use std::path::PathBuf;

mod math;
mod utilities;

/// Transformer model

pub struct Config {
    /// transformer dimension
    dim: usize,
    /// for ffn layers
    hidden_dim: usize,
    /// number of layers
    n_layers: usize,
    /// number of query heads
    n_heads: usize,
    /// number of key/value heads (can be < query heads because of multiquery)
    n_kv_heads: usize,
    /// vocabulary size, usually 256 (byte-level)
    pub vocab_size: usize,
    /// max sequence length
    pub seq_len: usize,
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

pub struct Transformer {
    /// the hyperparameters of the architecture (the blueprint)
    pub config: Config,
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

impl Transformer {
    pub(crate) fn build_transformer(checkpoint_path: &PathBuf) -> anyhow::Result<Self> {
        todo!()
    }

    pub(crate) fn generate(&self, p0: &Tokenizer, p1: &Sampler, p2: &String, p3: usize) -> anyhow::Result<()> {
        todo!()
    }

    pub(crate) fn chat(&self, p0: &Tokenizer, p1: &Sampler, p2: &String, p3: &Option<String>, p4: usize) -> anyhow::Result<()> {
        todo!()
    }
}


impl TransformerWeights {
    fn memory_map_weights(&mut self, config: &Config, ptr: &mut Vec<f32>, shared_weights: i32) {
        todo!("memory_map_weights")
    }
}

impl RunState {
    /// constructor
    fn malloc_run_state(config: &Config) -> RunState {
        let kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;
        let dim = config.dim;
        let hidden_dim = config.hidden_dim;
        RunState {
            x: Vec::with_capacity(dim),
            xb: Vec::with_capacity(dim),
            xb2: Vec::with_capacity(dim),
            hb: Vec::with_capacity(hidden_dim),
            hb2: Vec::with_capacity(hidden_dim),
            q: Vec::with_capacity(dim),
            k: vec![],
            att: Vec::with_capacity(config.n_heads * config.seq_len),
            logits: Vec::with_capacity(config.vocab_size),
            v: vec![],
            key_cache: Vec::with_capacity(config.n_layers * config.seq_len * kv_dim),
            value_cache: Vec::with_capacity(config.n_layers * config.seq_len * kv_dim),
        }
    }

    // `RunState::free_run_state` is implicit by `drop`
}


// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

pub struct Tokenizer {
    /// the vocabulary of the tokenizer
    vocab: Vec<String>,
    /// the scores of the vocabulary
    vocab_scores: Vec<f32>,
    /// the sorted vocabulary
    sorted_vocab: Box<TokenIndex>,
    /// the size of the vocabulary
    vocab_size: usize,
    /// the maximum token length
    max_token_length: usize,
    /// the byte pieces
    byte_pieces: Vec<u8>,
}

impl Tokenizer {
    pub(crate) fn build_tokenizer(tokenizer_path: &PathBuf, vocab_size: usize) -> anyhow::Result<Self> {
        todo!()
    }
}

struct TokenIndex {
    str: String,
    id: i32,
}


// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

/*
*/

pub struct ProbIndex {
    prob: f32,
    index: i32,
}

pub struct Sampler {
    /// the vocabulary size
    vocab_size: usize,
    /// buffer used in top-p sampling
    probindex: Box<ProbIndex>,
    /// the temperature of the sampling
    temperature: f32,
    /// the top-p value
    topp: f32,
    /// the random number generator seed
    rng_seed: u64,
}

impl Sampler {
    pub(crate) fn build_sampler(vocab_size: usize, temperature: f32, topp: f32, rng_seed: u64) -> anyhow::Result<Self> {
        todo!()
    }
}
