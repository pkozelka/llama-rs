use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use byteorder::{LittleEndian, ReadBytesExt};

use crate::run::sampler::Sampler;
use crate::run::tokenizer::Tokenizer;

mod math;
mod utilities;
mod forward;
pub mod tokenizer;
pub mod sampler;

/// Transformer model

#[derive(Default,Debug)]
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
    k_index: usize,
    /// value (dim,)
    v_index: usize,
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
    _fd: i32,
    /// memory mapped data pointer
    _data: Vec<f32>,
    /// size of the checkpoint file in bytes
    _file_size: i32,
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
        self.config = Config::read_config(&mut reader)?;
        eprintln!("config: {:?}", self.config);
        // ORIGINAL: memory map the Transformer weights into the data pointer
        //     *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
        // INSTEAD: we just read the weights, as mmap is not easily available in Rust
        self.weights.read_weights(&mut reader, &self.config)?;
        Ok(())
    }

    pub(crate) fn generate(&mut self, tokenizer: &mut Tokenizer, sampler: &mut Sampler, prompt: &str, steps: usize) -> anyhow::Result<()> {
        let prompt_tokens = tokenizer.encode(prompt, true, false)?;
        prompt_tokens.iter().enumerate().for_each(|(i, t)| eprintln!("prompt_tokens[{i}]={t}"));

        let mut start = 0;
        let mut token = prompt_tokens[0] as i32;
        let mut pos = 0;
        while pos < steps {
            // eprintln!("generate: pos: {}", pos);
            // forward the transformer to get logits for the next token
            self.forward(token as usize, pos)?;
            // advance the state machine
            let next = if pos + 1 < prompt_tokens.len() {
                // if we are still processing the input prompt, force the next prompt token
                prompt_tokens[pos + 1] as i32
            } else {
                // otherwise sample the next token from the logits
                sampler.sample(&mut self.state.logits)
            };
            pos += 1;
            // data-dependent terminating condition: the BOS (=1) token delimits sequences
            if next == 1 { break; }
            // print the token as string, decode it with the Tokenizer object
            let piece = tokenizer.decode(token, next);
            eprintln!("pos:{pos:3} token: {token:5} next: {next:5} piece: '{piece}'");
            utilities::safe_printf(&piece); // same as printf("%s", piece), but skips "unsafe" bytes
            token = next;
            // init the timer here because the first iteration can be slower
            if start == 0 { start = utilities::time_in_ms(); }
        }
        println!();
        // report achieved tok/s (pos-1 because the timer starts after first iteration)
        if pos > 1 {
            let end = utilities::time_in_ms();
            eprintln!("achieved tok/s: {}", (pos - 1) as f64 / (end - start) as f64 * 1000.0);
        }
        Ok(())
    }

    pub(crate) fn chat(&self, _tokenizer: &Tokenizer, _sampler: &Sampler, _cli_user_prompt: &String, _cli_system_prompt: &Option<String>, _steps: usize) -> anyhow::Result<()> {
        todo!()
    }
}


impl TransformerWeights {
    fn _memory_map_weights(&mut self, _config: &Config, _ptr: &mut Vec<f32>, _shared_weights: i32) {
        unimplemented!("memory mapping is not easily available in Rust, so we read the weights into memory")
    }

    /// currently we have this instead of memory_map_weights()
    fn read_weights(&mut self, reader: &mut BufReader<File>, p: &Config) -> anyhow::Result<()> {
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

        // eprintln!("shared_weights: {}", p.shared_weights);
        let wcls = if p.shared_weights {
            token_embedding_table.clone()
        } else {
            reader.seek_relative((p.seq_len * head_size / 2) as i64)?; // skip what used to be freq_cis_real (for RoPE)
            reader.seek_relative((p.seq_len * head_size / 2) as i64)?; // skip what used to be freq_cis_imag (for RoPE)
            utilities::read_f32_table(reader, p.vocab_size, p.dim)?
        };
        // eprintln!("wcls.len={}", wcls.len());

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
        // eprintln!("malloc_run_state(config={:?})", config);
        let kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;
        let dim = config.dim;
        let hidden_dim = config.hidden_dim;
        let key_cache = vec![0.0; config.n_layers * config.seq_len * kv_dim];
        let value_cache = vec![0.0; config.n_layers * config.seq_len * kv_dim];
        RunState {
            x: vec![0.0; dim],
            xb: vec![0.0; dim],
            xb2: vec![0.0; dim],
            hb: vec![0.0; hidden_dim],
            hb2: vec![0.0; hidden_dim],
            q: vec![0.0; dim],
            k_index: 0,
            v_index: 0,
            att: vec![0.0; config.n_heads * config.seq_len],
            logits: vec![0.0; config.vocab_size],
            key_cache: key_cache,
            value_cache: value_cache,
        }
    }

    // `RunState::free_run_state` is implicit by `drop`
}
