//! # LLAMA2-RS
//! This is simple rewrite of https://github.com/karpathy/llama2.c/blob/master/run.c
//! Minimum Rust fanciness used.
//!
use std::path::PathBuf;
use std::str::FromStr;

use clap::Parser;

use crate::llama2::sampler::Sampler;
use crate::llama2::tokenizer::Tokenizer;
use crate::llama2::Transformer;

mod llama2;

fn main() -> anyhow::Result<()> {
    env_logger::builder()
        // .filter_level(log::LevelFilter::Debug)
        .init();
    let mut args = Cli::parse();
    args.param_validation_overrides();

    // build the Transformer via the model .bin file
    let transformer = Transformer::new(&args.checkpoint_path)?;
    if args.steps == 0 || args.steps > transformer.config.seq_len {
        // override to ~max length
        args.steps = transformer.config.seq_len;
    }

    let args = args; // make args immutable

    // build the Tokenizer via the tokenizer .bin file
    let tokenizer = Tokenizer::build_tokenizer(&args.tokenizer_path, transformer.config.vocab_size)?;

    // build the Sampler
    let sampler = Sampler::build_sampler(transformer.config.vocab_size, args.temperature, args.topp, args.rng_seed)?;

    // run!
    match args.mode {
        Mode::Generate => {
            transformer.generate(&tokenizer, &sampler, &args.prompt, args.steps)?;
        }
        Mode::Chat => {
            transformer.chat(&tokenizer, &sampler, &args.prompt, &args.system_prompt, args.steps)?;
        }
    }

    Ok(())
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Cli {
    /// e.g. out/model.bin
    checkpoint_path: PathBuf,
    ///
    #[arg(short='z', default_value = "downloads/llama2c-tokenizer.bin")]
    tokenizer_path: PathBuf,
    /// 0.0 = greedy deterministic. 1.0 = original. don't set higher
    #[arg(short, default_value = "1.0")]
    temperature: f32,
    /// top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    #[arg(short='p', default_value = "0.9")]
    topp: f32,
    /// number of steps to run for
    #[arg(short='n', default_value = "256")]
    steps: usize,
    /// prompt string
    #[arg(short='i', default_value = "0")]
    prompt: String,
    /// seed rng with time by default
    #[arg(short='s', default_value = "0")]
    rng_seed: u64,
    /// generate|chat
    #[arg(short='m', default_value = "generate")]
    mode: Mode,
    /// the (optional) system prompt to use in chat mode
    #[arg(short='y')]
    system_prompt: Option<String>,
}

impl Cli {
    fn param_validation_overrides(&mut self) {
        if self.rng_seed == 0 {
            self.rng_seed = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();
            log::warn!("seed not set, using {}", self.rng_seed)
        }
        if self.temperature < 0.0 {
            log::warn!("stripping temperature to 0.0");
        }
        if self.topp < 0.0 || 1.0 < self.topp {
            self.topp = 0.9;
            log::warn!("setting topp to {}", self.topp);
        }
    }
}

#[derive(Clone, Parser, Debug)]
enum Mode {
    Generate,
    Chat,
}

impl FromStr for Mode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "generate" => Ok(Mode::Generate),
            "chat" => Ok(Mode::Chat),
            _ => Err("mode must be 'generate' or 'chat'".to_string()),
        }
    }
}
