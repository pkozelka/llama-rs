use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

use byteorder::{LittleEndian, ReadBytesExt};

mod math;
mod utilities;

/// Transformer model

#[derive(Default)]
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

    shared_weights: bool,
}

#[derive(Default)]
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
#[derive(Default)]
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

#[derive(Default)]
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

impl Config {
    pub(crate) fn read_config(reader: &mut BufReader<File>) -> anyhow::Result<Self> {
        let dim = reader.read_i32::<LittleEndian>()?;
        let hidden_dim = reader.read_i32::<LittleEndian>()?;
        let n_layers = reader.read_i32::<LittleEndian>()?;
        let n_heads = reader.read_i32::<LittleEndian>()?;
        let n_kv_heads = reader.read_i32::<LittleEndian>()?;
        let vocab_size = reader.read_i32::<LittleEndian>()?;
        let seq_len = reader.read_i32::<LittleEndian>()?;

        // negative vocab size is hacky way of signaling unshared weights. bit yikes.
        let shared_weights = vocab_size > 0;
        let vocab_size = vocab_size.abs();

        Ok(Self {
            dim: dim as usize,
            hidden_dim: hidden_dim as usize,
            n_layers: n_layers as usize,
            n_heads: n_heads as usize,
            n_kv_heads: n_kv_heads as usize,
            vocab_size: vocab_size as usize,
            seq_len: seq_len as usize,
            shared_weights,
        })
    }
}

impl Transformer {
    pub(crate) fn build_transformer(checkpoint_path: &PathBuf) -> anyhow::Result<Self> {
        let mut transformer = Self::default();
        transformer.read_checkpoint(checkpoint_path)?;
        transformer.state = RunState::malloc_run_state(&transformer.config);
        Ok(transformer)
    }

    fn read_checkpoint(&mut self, checkpoint_path: &PathBuf) -> anyhow::Result<()> {
        let file = File::open(checkpoint_path)?;
        let mut reader = BufReader::new(file);
        // read in the config header from reader
        let config = Config::read_config(&mut reader)?;
        // ORIGINAL: memory map the Transformer weights into the data pointer
        //     *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
        // INSTEAD: we just read the weights, as mmap is not easily available in Rust
        self.weights.read_weights(&mut reader, &config)?;
        Ok(())
    }

    pub(crate) fn generate(&self, p0: &Tokenizer, p1: &Sampler, p2: &String, p3: usize) -> anyhow::Result<()> {
        todo!()
    }

    pub(crate) fn chat(&self, p0: &Tokenizer, p1: &Sampler, p2: &String, p3: &Option<String>, p4: usize) -> anyhow::Result<()> {
        todo!()
    }
}


impl TransformerWeights {
    fn _memory_map_weights(&mut self, config: &Config, ptr: &mut Vec<f32>, shared_weights: i32) {
        unimplemented!("memory mapping is not easily available in Rust, so we read the weights into memory")
    }

    fn read_weights(&mut self, reader: &mut BufReader<File>, config: &Config) -> anyhow::Result<()> {
        let dim = config.dim;
        let hidden_dim = config.hidden_dim;
        let n_layers = config.n_layers;
        let n_heads = config.n_heads;
        let n_kv_heads = config.n_kv_heads;

        // read in the token embedding table
        let token_embedding_table = utilities::read_f32_table(reader, config.vocab_size, dim)?;
        // read in the rmsnorm weights
        let rms_att_weight = utilities::read_f32_table(reader, n_layers, dim)?;
        let rms_ffn_weight = utilities::read_f32_table(reader, n_layers, dim)?;
        // read in the q, k, v, and o weights
        let wq = utilities::read_f32_table(reader, n_layers, dim * n_heads)?;
        let wk = utilities::read_f32_table(reader, n_layers, dim * n_kv_heads)?;
        let wv = utilities::read_f32_table(reader, n_layers, dim * n_kv_heads)?;
        let wo = utilities::read_f32_table(reader, n_layers, n_heads * dim)?;
        // read in the ffn weights
        let w1 = utilities::read_f32_table(reader, n_layers, hidden_dim * dim)?;
        let w2 = utilities::read_f32_table(reader, n_layers, dim * hidden_dim)?;
        let w3 = utilities::read_f32_table(reader, n_layers, hidden_dim * dim)?;
        // read in the final rmsnorm weights
        let rms_final_weight = utilities::read_f32_table(reader, dim, 1)?;
        // read in the classifier weights
        let wcls = utilities::read_f32_table(reader, config.vocab_size, dim)?;

        // assign the weights
        self.token_embedding_table = token_embedding_table;
        self.rms_att_weight = rms_att_weight;
        self.rms_ffn_weight = rms_ffn_weight;
        self.wq = wq;
        self.wk = wk;
        self.wv = wv;
        self.wo = wo;
        self.w1 = w1;
        self.w2 = w2;
        self.w3 = w3;
        self.rms_final_weight = rms_final_weight;
        self.wcls = wcls;

        Ok(())
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
