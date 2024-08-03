use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use llama_rs::config::Config;
use llama_rs::dirty_dbg;
use weights::TransformerWeights;
use crate::llama2::sampler::Sampler;
use crate::llama2::tokenizer::Tokenizer;

mod math;
mod utilities;
mod forward;
pub mod tokenizer;
pub mod sampler;

mod weights;

mod chat;

/// current wave of activations
#[derive(Default)]
pub struct RunState {
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

pub struct Transformer {
    /// the hyperparameters of the architecture (the blueprint)
    pub config: Config,
    /// the weights of the model
    weights: TransformerWeights,
}

impl Transformer {
    /// Initializes the Transformer object.
    ///
    /// This function corresponds to these original functions in the `llama2.c` code:
    /// - build_transformer
    /// - read_checkpoint
    /// - malloc_run_state
    pub fn new(checkpoint_path: &Path) -> anyhow::Result<Self> {
        let file = File::open(checkpoint_path)?;
        let mut reader = BufReader::new(file);
        // read in the config header from reader
        let config = Config::read_config(&mut reader)?;
        dirty_dbg!("config: {:?}", config);
        let weights = TransformerWeights::read_weights(&mut reader, &config)?;
        Ok(Self{ config, weights })
    }

    pub(crate) fn generate(&self, tokenizer: &Tokenizer, sampler: &Sampler, prompt: &str, steps: usize) -> anyhow::Result<()> {
        let  mut runstate = RunState::malloc_run_state(&self.config);

        let prompt_tokens = tokenizer.encode(prompt, true, false)?;
        prompt_tokens.iter().enumerate().for_each(|(i, t)| dirty_dbg!("prompt_tokens[{i}]={t}"));

        let mut start = 0;
        let mut token = prompt_tokens[0] as i32;
        let mut pos = 0;
        while pos < steps {
            // log_debug!("generate: pos: {}", pos);
            // forward the transformer to get logits for the next token
            runstate.forward(&self, token as usize, pos)?;
            // advance the state machine
            pos += 1;
            let next = if pos < prompt_tokens.len() {
                // if we are still processing the input prompt, force the next prompt token
                prompt_tokens[pos] as i32
            } else {
                // otherwise sample the next token from the logits
                sampler.sample(&mut runstate.logits)
            };
            // data-dependent terminating condition: the BOS (=1) token delimits sequences
            if next == 1 { break; }
            // print the token as string, decode it with the Tokenizer object
            let piece = tokenizer.decode(token, next);
            dirty_dbg!("pos:{pos:3} token: {token:5} next: {next:5} piece: '{piece}'");
            utilities::safe_printf(&piece); // same as printf("%s", piece), but skips "unsafe" bytes
            token = next;
            // init the timer here because the first iteration can be slower
            if start == 0 { start = utilities::time_in_ms(); }
        }
        println!();
        // report achieved tok/s (pos-1 because the timer starts after first iteration)
        if pos > 1 {
            let end = utilities::time_in_ms();
            eprintln!("\nachieved tok/s: {}", (pos - 1) as f64 / (end - start) as f64 * 1000.0);
        }
        Ok(())
    }
}


impl RunState {
    /// constructor
    pub(crate) fn malloc_run_state(config: &Config) -> RunState {
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
