use llama_rs::config::Config;
use crate::llama2_q::q80::QuantizedTensor;

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

}