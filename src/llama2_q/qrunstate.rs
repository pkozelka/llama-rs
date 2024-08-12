use llama_rs::config::Config;
use llama_rs::dirty_dbg;
use llama_rs::math::{matmul, rmsnorm, rmsnorm_inplace, softmax};
use crate::llama2_q::q80::QuantizedTensor;
use crate::llama2_q::QTransformer;

/// current wave of activations
#[derive(Default)]
pub struct QRunState {
    /// activation at current time stamp (dim,)
    x: Vec<f32>,
    /// same, but inside a residual branch (dim,)
    xb: Vec<f32>,
    /// an additional buffer just for convenience (dim,)
    xb2: Vec<f32>,
    /// buffer for hidden dimension in the ffn (hidden_dim,)
    hb: Vec<f32>,
    /// buffer for hidden dimension in the ffn (hidden_dim,)
    hb2: Vec<f32>,
    // quantized x (dim,)
    xq: QuantizedTensor,
    // quantized hb (hidden_dim,)
    hq: QuantizedTensor,
    /// query (dim,)
    q: Vec<f32>,
    /// key (dim,)
    k_index: usize,
    /// value (dim,)
    v_index: usize,
    /// buffer for scores/attention values (n_heads, seq_len)
    att: Vec<f32>,
    /// output logits
    pub(crate) logits: Vec<f32>,

    // kv cache

    /// (layer, seq_len, dim)
    key_cache: Vec<f32>,
    /// (layer, seq_len, dim)
    value_cache: Vec<f32>,
}


impl QRunState {
    pub(crate) fn malloc_run_state(config: &Config) -> Self {
        let kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;
        let dim = config.dim;
        let hidden_dim = config.hidden_dim;
        let key_cache = vec![0.0; config.n_layers * config.seq_len * kv_dim];
        let value_cache = vec![0.0; config.n_layers * config.seq_len * kv_dim];
        Self {
            x: vec![0.0; dim],
            xb: vec![0.0; dim],
            xb2: vec![0.0; dim],
            hb: vec![0.0; hidden_dim],
            hb2: vec![0.0; hidden_dim],
            q: vec![0.0; dim],
            xq: QuantizedTensor::new(dim),
            hq: QuantizedTensor::new(hidden_dim),
            k_index: 0,
            v_index: 0,
            att: vec![0.0; config.n_heads * config.seq_len],
            logits: vec![0.0; config.vocab_size],
            key_cache: key_cache,
            value_cache: value_cache,
        }
    }

    // `QRunState::free_run_state` is implicit by `drop`

    pub fn forward(&mut self, transformer: &QTransformer, token: usize, pos: usize) -> anyhow::Result<()> {
        dirty_dbg!("forward(token={}, pos={})", token, pos);
        let p = &transformer.config;
        let w = &transformer.weights;
        let s = self;
        let x = &mut s.x;
        let dim = p.dim;
        let kv_dim = (p.dim * p.n_kv_heads) / p.n_heads;
        let kv_mul = p.n_heads / p.n_kv_heads;
        let head_size = dim / p.n_heads;

        // copy the token embedding into x
        let content_row = &w.token_embedding_table[token * dim..];
        let content_row = &content_row[..dim];

        x.copy_from_slice(content_row);

        // forward all the layers
        for l in 0..p.n_layers {
            // attention rmsnorm
            let weight = &w.rms_att_weight[l * dim..];
            let weight = &weight[..dim];
            rmsnorm(&mut s.xb, x, weight);
            dirty_dbg!("L{l:02} A) xb[0] = {:1.6}", s.xb[0]);

            // key and value point to the kv cache
            let loff = l * p.seq_len * kv_dim; // kv cache layer offset for convenience
            s.k_index = loff + pos * kv_dim;
            s.v_index = loff + pos * kv_dim;

            // qkv matmuls for this position
            s.xq.quantize(&s.xb);
            QuantizedTensor::matmul(&mut s.q[..dim], &s.xq, &w.wq[l]);
            let k = &mut s.key_cache[s.k_index..][..kv_dim];
            QuantizedTensor::matmul(k, &s.xq, &w.wk[l * dim * kv_dim..][0]);
            let v = &mut s.value_cache[s.v_index..][..kv_dim];
            QuantizedTensor::matmul(v, &s.xq, &w.wv[l * dim * kv_dim..][0]);
            dirty_dbg!("L{l:02} B) (q,k,v)[0] = ({:1.6},{:1.6},{:1.6})", s.q[0], k[0], v[0]);

            // RoPE relative positional encoding: complex-valued rotate q and k in each head
            for i in (0..dim).step_by(2) {
                let head_dim = i % head_size;
                let freq = 1.0 / 10000.0f32.powf(head_dim as f32 / head_size as f32);
                let val = pos as f32 * freq;
                let fcr = val.cos();
                let fci = val.sin();
                let rotn = if i < kv_dim { 2 } else { 1 }; // how many vectors? 2 = q & k, 1 = q only
                for v in 0..rotn {
                    let vec = if v == 0 {
                        s.q.as_mut_slice()
                    } else {
                        // k
                        &mut s.key_cache[s.k_index..]
                    };  // the vector to rotate (query or key)
                    let v0 = vec[i];
                    let v1 = vec[i + 1];
                    vec[i] = v0 * fcr - v1 * fci;
                    vec[i + 1] = v0 * fci + v1 * fcr;
                }
            }
            dirty_dbg!("L{l:02} C) s.q[0..4] = ({:1.6},{:1.6},{:1.6},{:1.6})", s.q[0], s.q[1], s.q[2], s.q[3]);

            // multihead attention. iterate over all heads
            // #pragma omp parallel for private(h) //TODO: parallelize this with rayon
            for h in 0..p.n_heads {
                // get the query vector for this head
                let q = &s.q[h * head_size..];
                dirty_dbg!("L{l:02}/h{h} Ca) q[0,1,2,3] = ({:1.6},{:1.6},{:1.6},{:1.6})", q[0], q[1], q[2], q[3]);
                // attention scores for this head
                let att = &mut s.att[h * p.seq_len..];
                // iterate over all timesteps, including the current one
                for t in 0..=pos {
                    // get the key vector for this head and at this timestep
                    let k = &s.key_cache[loff + t * kv_dim + (h / kv_mul) * head_size..];
                    dirty_dbg!("L{l:02}/h{h}/t{t} CbA) k[0,1,2,3] = ({:1.6},{:1.6},{:1.6},{:1.6})", k[0], k[1], k[2], k[3]);
                    // calculate the attention score as the dot product of q and k
                    let mut score = 0.0;
                    for i in 0..head_size {
                        score += q[i] * k[i];
                    }
                    score /= (head_size as f32).sqrt();
                    // save the score to the attention buffer
                    att[t] = score;
                }
                dirty_dbg!("L{l:02}/h{h} Cc) att[0,1,2,3] = ({:1.6},{:1.6},{:1.6},{:1.6})", att[0], att[1], att[2], att[3]);

                // softmax the scores to get attention weights, from 0..pos inclusively
                softmax(&mut att[..=pos]);

                // weighted sum of the values, store back into xb
                let xb = &mut s.xb[h * head_size..];
                xb.iter_mut().for_each(|x| *x = 0.0);
                for t in 0..=pos as usize {
                    // get the value vector for this head and at this timestep
                    let v = &s.value_cache[loff + t * kv_dim + (h / kv_mul) * head_size..];
                    // get the attention weight for this timestep
                    let a = att[t];
                    // accumulate the weighted value into xb
                    for i in 0..head_size {
                        xb[i] += a * v[i];
                    }
                }
            }
            dirty_dbg!("L{l:02} D) xb[0,1,2,3] = ({:1.6},{:1.6},{:1.6},{:1.6})", s.xb[0], s.xb[1], s.xb[2], s.xb[3]);

            // final matmul to get the output of the attention
            s.xq.quantize(&s.xb);
            QuantizedTensor::matmul(&mut s.xb2[..dim], &s.xq, &w.wo[l * dim * dim..][0]);
            dirty_dbg!("L{l:02} E) xb2[0,1,2,3] = ({:1.6},{:1.6},{:1.6},{:1.6})", s.xb2[0], s.xb2[1], s.xb2[2], s.xb2[3]);

            // residual connection back into x
            for i in 0..dim {
                x[i] += s.xb2[i];
            }
            dirty_dbg!("L{l:02} F) x[0,1,2,3] = ({:1.6},{:1.6},{:1.6},{:1.6})", x[0], x[1], x[2], x[3]);

            // ffn rmsnorm
            rmsnorm(&mut s.xb, x, &w.rms_ffn_weight[l * dim..]);
            dirty_dbg!("L{l:02} G) xb[0,1,2,3] = ({:1.6},{:1.6},{:1.6},{:1.6})", s.xb[0], s.xb[1], s.xb[2], s.xb[3]);

            // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
            // first calculate self.w1(x) and self.w3(x)
            let hidden_dim = p.hidden_dim;
            QuantizedTensor::matmul(&mut s.hb[..hidden_dim], &s.xq, &w.w1[l * dim * hidden_dim..][0]);
            dirty_dbg!("L{l:02} H) hb[0,1,2,3] = ({:1.6},{:1.6},{:1.6},{:1.6})", s.hb[0], s.hb[1], s.hb[2], s.hb[3]);
            QuantizedTensor::matmul(&mut s.hb2[..hidden_dim], &s.xq, &w.w3[l * dim * hidden_dim..][0]);
            dirty_dbg!("L{l:02} I) hb2[0,1,2,3] = ({:1.6},{:1.6},{:1.6},{:1.6})", s.hb2[0], s.hb2[1], s.hb2[2], s.hb2[3]);

            // SwiGLU non-linearity
            for i in 0..hidden_dim {
                let mut val = s.hb[i];
                // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
                val *= 1.0 / (1.0 + (-val).exp());
                // elementwise multiply with w3(x)
                val *= s.hb2[i];
                s.hb[i] = val;
            }
            dirty_dbg!("L{l:02} J) hb[0,1,2,3] = ({:1.6},{:1.6},{:1.6},{:1.6})", s.hb[0], s.hb[1], s.hb[2], s.hb[3]);

            // final matmul to get the output of the ffn
            s.hq.quantize(&s.hb);
            QuantizedTensor::matmul(&mut s.xb[..dim], &s.hq, &w.w2[l * dim * hidden_dim..][0]);
            dirty_dbg!("L{l:02} K) xb[0,1,2,3] = ({:1.6},{:1.6},{:1.6},{:1.6})", s.xb[0], s.xb[1], s.xb[2], s.xb[3]);

            // residual connection
            for i in 0..dim {
                x[i] += s.xb[i];
            }

            dirty_dbg!("L{l:02} Z) xb[0,1,2,3] = ({:1.6},{:1.6},{:1.6},{:1.6})", s.xb[0], s.xb[1], s.xb[2], s.xb[3]);
        }

        // final rmsnorm
        rmsnorm_inplace(x, &w.rms_final_weight);
        dirty_dbg!("Final_A) x[0,1,2,3,4] = ({:1.6},{:1.6},{:1.6},{:1.6},{:1.6})", x[0], x[1], x[2], x[3], x[4]);

        // classifier into logits
        s.xq.quantize(&x);
        QuantizedTensor::matmul(&mut s.logits[..p.vocab_size], &s.xq, &w.wcls);
        Ok(())
    }

}