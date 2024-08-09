use llama_rs::config::Config;
use std::io::BufReader;
use std::fs::File;
use llama_rs::utilities;

#[derive(Default)]
pub struct TransformerWeights {
    // token embedding table
    /// (vocab_size, dim)
    pub(crate) token_embedding_table: Vec<f32>,

    // weights for rmsnorms
    /// (layer, dim) rmsnorm weights
    pub(crate) rms_att_weight: Vec<f32>,
    /// (layer, dim)
    pub(crate) rms_ffn_weight: Vec<f32>,
    /// (layer, dim, n_heads * head_size)
    pub(crate) wq: Vec<f32>,
    /// (layer, dim, n_kv_heads * head_size)
    pub(crate) wk: Vec<f32>,
    /// (layer, dim, n_kv_heads * head_size)
    pub(crate) wv: Vec<f32>,
    /// (layer, n_heads * head_size, dim)
    pub(crate) wo: Vec<f32>,

    // weights for ffn
    /// (layer, hidden_dim, dim)
    pub(crate) w1: Vec<f32>,
    /// (layer, dim, hidden_dim)
    pub(crate) w2: Vec<f32>,
    /// (layer, hidden_dim, dim)
    pub(crate) w3: Vec<f32>,

    // final rmsnorm
    /// (dim,)
    pub(crate) rms_final_weight: Vec<f32>,
    /// (optional) classifier weights for the logits, on the last layer
    pub(crate) wcls: Vec<f32>,
}

impl TransformerWeights {

    /// ORIGINAL: memory map the Transformer weights into the data pointer
    ///     *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    /// INSTEAD: we just read the weights, as mmap is not easily available in Rust
    ///
    /// TODO: consider using this for mmap stuff: https://docs.rs/memmap/latest/memmap/struct.Mmap.html
    ///
    /// OTOH, I am not sure if mmap would be such a big win here
    fn _memory_map_weights(&mut self, _config: &Config, _ptr: &mut Vec<f32>, _shared_weights: i32) {
        unimplemented!("memory mapping is not easily available in Rust, so we read the weights into memory")
    }

    /// currently we have this instead of memory_map_weights()
    pub(crate) fn read_weights(reader: &mut BufReader<File>, p: &Config) -> anyhow::Result<TransformerWeights> {
        let head_size = p.dim / p.n_heads;
        let hidden_dim = p.hidden_dim;
        let n_layers = p.n_layers;

        let token_embedding_table = utilities::read_f32_table(reader, p.vocab_size, p.dim)?;

        let rms_att_weight = utilities::read_f32_table(reader, n_layers, p.dim)?;

        let wq = utilities::read_f32_table(reader, n_layers, p.dim * (p.n_heads * head_size))?;
        let wk = utilities::read_f32_table(reader, n_layers, p.dim * (p.n_kv_heads * head_size))?;
        let wv = utilities::read_f32_table(reader, n_layers, p.dim * (p.n_kv_heads * head_size))?;
        let wo = utilities::read_f32_table(reader, n_layers, p.dim * (p.n_heads * head_size))?;
        //
        let rms_ffn_weight = utilities::read_f32_table(reader, n_layers, p.dim)?;

        let w1 = utilities::read_f32_table(reader, n_layers, p.dim * hidden_dim)?;
        let w2 = utilities::read_f32_table(reader, n_layers, p.dim * hidden_dim)?;
        let w3 = utilities::read_f32_table(reader, n_layers, p.dim * hidden_dim)?;

        let rms_final_weight = utilities::read_f32_table(reader, 1, p.dim)?;

        let wcls = if p.shared_weights {
            token_embedding_table.clone()
        } else {
            reader.seek_relative((p.seq_len * head_size / 2) as i64)?; // skip what used to be freq_cis_real (for RoPE)
            reader.seek_relative((p.seq_len * head_size / 2) as i64)?; // skip what used to be freq_cis_imag (for RoPE)
            utilities::read_f32_table(reader, p.vocab_size, p.dim)?
        };

        Ok(TransformerWeights {
            token_embedding_table,
            rms_att_weight,
            wq,
            wk,
            wv,
            wo,
            rms_ffn_weight,
            w1,
            w2,
            w3,
            rms_final_weight,
            wcls,
        })
    }
}