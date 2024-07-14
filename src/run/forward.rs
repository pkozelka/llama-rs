use crate::run::math::{matmul, rmsnorm, rmsnorm_inplace, softmax};
use crate::run::Transformer;

impl Transformer {
    pub fn forward<'a>(&'a mut self, token: usize, pos: usize) -> anyhow::Result<&'a [f32]> {
        log::debug!("forward(token={}, pos={})", token, pos);
        let p = &self.config;
        let w = &self.weights;
        let s = &mut self.state;
        let x = &mut s.x;
        let dim = p.dim;
        let kv_dim = (p.dim * p.n_kv_heads) / p.n_heads;
        let kv_mul = p.n_heads / p.n_kv_heads;
        let hidden_dim = p.hidden_dim;
        let head_size = dim / p.n_heads;

        // copy the token embedding into x
        let content_row = &w.token_embedding_table[token * dim..];
        let content_row = &content_row[..dim];
        *x = content_row.to_vec();

        // forward all the layers
        for l in 0..p.n_layers {
            // attention rmsnorm
            let weight = &w.rms_att_weight[l * dim..];
            let weight = &weight[..dim];
            rmsnorm(&mut s.xb, x, weight);

            // key and value point to the kv cache
            let loff = l * p.seq_len * kv_dim; // kv cache layer offset for convenience
            s.k = s.key_cache[loff + pos * kv_dim..].to_vec(); //TODO: avoid this copy
            s.v = s.value_cache[loff + pos * kv_dim..].to_vec(); //TODO: avoid this copy

            // qkv matmuls for this position
            matmul(&mut s.q, &s.xb, &w.wq[l * dim * dim..], dim, dim);
            matmul(&mut s.k, &s.xb, &w.wk[l * dim * kv_dim..], dim, kv_dim);
            matmul(&mut s.v, &s.xb, &w.wv[l * dim * kv_dim..], dim, kv_dim);

            // RoPE relative positional encoding: complex-valued rotate q and k in each head
            for i in (0..dim).step_by(2) {
                let head_dim = i % head_size;
                let freq = 1.0 / 10000.0f32.powf(head_dim as f32 / head_size as f32);
                let val = pos as f32 * freq;
                let fcr = val.cos();
                let fci = val.sin();
                let rotn = if i < kv_dim { 2 } else { 1 }; // how many vectors? 2 = q & k, 1 = q only
                for v in 0..rotn {
                    let vec = if v == 0 { &mut s.q } else { &mut s.k };  // the vector to rotate (query or key)
                    let v0 = vec[i];
                    let v1 = vec[i + 1];
                    vec[i] = v0 * fcr - v1 * fci;
                    vec[i + 1] = v0 * fci + v1 * fcr;
                }
            }

            // multihead attention. iterate over all heads
            // #pragma omp parallel for private(h) //TODO: parallelize this with rayon
            for h in 0..p.n_heads {
                // get the query vector for this head
                let q = &s.q[h * head_size..];
                // attention scores for this head
                let att = &mut s.att[h * p.seq_len..];
                // iterate over all timesteps, including the current one
                for t in 0..=pos {
                    // get the key vector for this head and at this timestep
                    let k = &s.key_cache[loff + t * kv_dim + (h / kv_mul) * head_size..];
                    // calculate the attention score as the dot product of q and k
                    let mut score = 0.0;
                    for i in 0..head_size {
                        score += q[i] * k[i];
                    }
                    score /= (head_size as f32).sqrt();
                    // save the score to the attention buffer
                    att[t] = score;
                }

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

            // final matmul to get the output of the attention
            matmul(&mut s.xb2, &s.xb, &w.wo[l * dim * dim..], dim, dim);

            // residual connection back into x
            for i in 0..dim {
                x[i] += s.xb2[i];
            }

            // ffn rmsnorm
            rmsnorm(&mut s.xb, x, &w.rms_ffn_weight[l * dim..]);

            // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
            // first calculate self.w1(x) and self.w3(x)
            matmul(&mut s.hb, &s.xb, &w.w1[l * dim * hidden_dim..], dim, hidden_dim);
            matmul(&mut s.hb2, &s.xb, &w.w3[l * dim * hidden_dim..], dim, hidden_dim);

            // SwiGLU non-linearity
            for i in 0..hidden_dim {
                let mut val = s.hb[i];
                // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
                val *= 1.0 / (1.0 + (-val).exp());
                // elementwise multiply with w3(x)
                val *= s.hb2[i];
                s.hb[i] = val;
            }

            // final matmul to get the output of the ffn
            matmul(&mut s.xb, &s.hb, &w.w2[l * dim * hidden_dim..], hidden_dim, dim);

            // residual connection
            for i in 0..dim {
                x[i] += s.xb[i];
            }
        }

        // final rmsnorm
        rmsnorm_inplace(x, &w.rms_final_weight);

        // classifier into logits
        matmul(&mut s.logits, x, &w.wcls, p.dim, p.vocab_size);
        Ok(s.logits.as_slice())
    }
}
