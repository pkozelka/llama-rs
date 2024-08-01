use llama_rs::dirty_dbg;
use crate::llama2::math::{matmul, rmsnorm, rmsnorm_inplace, softmax};
use crate::llama2::Transformer;

impl Transformer {
    pub fn forward(&mut self, token: usize, pos: usize) -> anyhow::Result<()> {
        dirty_dbg!("forward(token={}, pos={})", token, pos);
        let p = &self.config;
        let w = &self.weights;
        let s = &mut self.state;
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
            matmul(&mut s.q, &s.xb, &w.wq[l * dim * dim..], dim, dim);
            let k = &mut s.key_cache[s.k_index..];
            matmul(k, &s.xb, &w.wk[l * dim * kv_dim..], dim, kv_dim);
            let v = &mut s.value_cache[s.v_index..];
            matmul(v, &s.xb, &w.wv[l * dim * kv_dim..], dim, kv_dim);
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
                let pos = pos as f32;
                let att = &mut s.att[h * p.seq_len..];
                let att = &mut att[..=pos as usize];
                softmax(att);

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
            matmul(&mut s.xb2, &s.xb, &w.wo[l * dim * dim..], dim, dim);
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
            matmul(&mut s.hb, &s.xb, &w.w1[l * dim * hidden_dim..], dim, hidden_dim);
            dirty_dbg!("L{l:02} H) hb[0,1,2,3] = ({:1.6},{:1.6},{:1.6},{:1.6})", s.hb[0], s.hb[1], s.hb[2], s.hb[3]);
            matmul(&mut s.hb2, &s.xb, &w.w3[l * dim * hidden_dim..], dim, hidden_dim);
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
            matmul(&mut s.xb, &s.hb, &w.w2[l * dim * hidden_dim..], hidden_dim, dim);
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
        matmul(&mut s.logits, x, &w.wcls, p.dim, p.vocab_size);
        Ok(())
    }
}