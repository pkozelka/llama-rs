// ----------------------------------------------------------------------------
// Transformer model

struct Config {
    /// transformer dimension
    dim: i32,
    /// for ffn layers
    hidden_dim: i32,
    /// number of layers
    n_layers: i32,
    /// number of query heads
    n_heads: i32,
    /// number of key/value heads (can be < query heads because of multiquery)
    n_kv_heads: i32,
    /// vocabulary size, usually 256 (byte-level)
    vocab_size: i32,
    /// max sequence length
    seq_len: i32,
}

struct TransformerWeights {
    /// token embedding table
    /// (vocab_size, dim)
    token_embedding_table: Vec<f32>,
    /// weights for rmsnorms
    /// (layer, dim) rmsnorm weights
    rms_att_weight: Vec<f32>,
    /// (layer, dim)
    rms_ffn_weight: Vec<f32>,
    /// (layer, dim, n_heads * head_size)
    wq: Vec<f32>,
    /// (layer, dim, n_kv_heads * head_size)
    wk: Vec<f32>,
    /// (layer, dim, n_kv_heads * head_size)
    wv: Vec<f32>,
    /// (layer, n_heads * head_size, dim)
    wo: Vec<f32>,
    /// weights for ffn
    /// (layer, hidden_dim, dim)
    w1: Vec<f32>,
    /// (layer, dim, hidden_dim)
    w2: Vec<f32>,
    /// (layer, hidden_dim, dim)
    w3: Vec<f32>,

    // final rmsnorm
    /// (dim,)
    rms_final_weight: Vec<f32>,
    /// (optional) classifier weights for the logits, on the last layer
    wcls: Vec<f32>,
}

/// current wave of activations
struct RunState {
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
    /// query (dim,)
    q: Vec<f32>,
    /// key (dim,)
    k: Vec<f32>,
    /// value (dim,)
    v: Vec<f32>,
    /// buffer for scores/attention values (n_heads, seq_len)
    att: Vec<f32>,
    /// output logits
    logits: Vec<f32>,

    // kv cache

    /// (layer, seq_len, dim)
    key_cache: Vec<f32>,
    /// (layer, seq_len, dim)
    value_cache: Vec<f32>,
}

struct Transformer {
    /// the hyperparameters of the architecture (the blueprint)
    config: Config,
    /// the weights of the model
    weights: TransformerWeights,
    /// buffers for the "wave" of activations in the forward pass
    state: RunState,
    /// some more state needed to properly clean up the memory mapping (sigh)
    fd: i32,
    /// memory mapped data pointer
    data: Vec<f32>,
    /// size of the checkpoint file in bytes
    file_size: i32,
}

/* TODO translate from C to Rust:

void memory_map_weights(TransformerWeights *w, Config* p, float* ptr, int shared_weights) {
    int head_size = p->dim / p->n_heads;
    // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
    unsigned long long n_layers = p->n_layers;
    w->token_embedding_table = ptr;
    ptr += p->vocab_size * p->dim;
    w->rms_att_weight = ptr;
    ptr += n_layers * p->dim;
    w->wq = ptr;
    ptr += n_layers * p->dim * (p->n_heads * head_size);
    w->wk = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wv = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wo = ptr;
    ptr += n_layers * (p->n_heads * head_size) * p->dim;
    w->rms_ffn_weight = ptr;
    ptr += n_layers * p->dim;
    w->w1 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->w2 = ptr;
    ptr += n_layers * p->hidden_dim * p->dim;
    w->w3 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->rms_final_weight = ptr;
    ptr += p->dim;
    ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_real (for RoPE)
    ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_imag (for RoPE)
    w->wcls = shared_weights ? w->token_embedding_table : ptr;
}
 */

impl TransformerWeights {
    fn memory_map_weights(&mut self, config: &Config, ptr: &mut Vec<f32>, shared_weights: i32) {
        todo!("memory_map_weights")
    }
}

impl RunState {
    fn new(config: &Config) -> RunState {
        let kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;
        let dim = config.dim as usize;
        let hidden_dim = config.hidden_dim as usize;
        RunState {
            x: Vec::with_capacity(dim),
            xb: Vec::with_capacity(dim),
            xb2: Vec::with_capacity(dim),
            hb: Vec::with_capacity(hidden_dim),
            hb2: Vec::with_capacity(hidden_dim),
            q: Vec::with_capacity(dim),
            k: vec![],
            att: Vec::with_capacity((config.n_heads * config.seq_len) as usize),
            logits: Vec::with_capacity(config.vocab_size as usize),
            v: vec![],
            key_cache: Vec::with_capacity((config.n_layers * config.seq_len * kv_dim) as usize),
            value_cache: Vec::with_capacity((config.n_layers * config.seq_len * kv_dim) as usize),
        }
    }
}

fn main() {
    println!("Hello, world!");
}
