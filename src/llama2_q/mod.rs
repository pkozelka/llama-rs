use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use llama_rs::{dirty_dbg, utilities};
use llama_rs::config::Config;
use llama_rs::sampler::Sampler;
use llama_rs::tokenizer::Tokenizer;

use crate::llama2_q::qrunstate::QRunState;
use crate::llama2_q::qweights::QTransformerWeights;

mod qweights;
mod qrunstate;
pub mod q80;

pub struct QTransformer {
    /// the hyperparameters of the architecture (the blueprint)
    pub config: Config,
    /// the weights of the model
    weights: QTransformerWeights,
}

impl QTransformer {
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
        let weights = QTransformerWeights::read_weights(&mut reader, &config)?;
        Ok(Self { config, weights })
    }

    pub(crate) fn generate(&self, tokenizer: &Tokenizer, sampler: &Sampler, prompt: &str, steps: usize) -> anyhow::Result<()> {
        let mut runstate = QRunState::malloc_run_state(&self.config);

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

    /// chat loop: comment from the original C code:
    /// > I manually inspected the tokens for a few chat conversations compared to
    /// > python reference and that seemed ok, but this was not thoroughly tested and
    /// > is not safely implemented, it's more a proof of concept atm.
    pub(crate) fn chat(&self, tokenizer: &Tokenizer, sampler: &Sampler, cli_user_prompt: &String, cli_system_prompt: &Option<String>, steps: usize) -> anyhow::Result<()> {
        let mut runstate = QRunState::malloc_run_state(&self.config);

        let mut prompt_tokens = Vec::new();
        let mut user_idx = 0;

        // start the main loop
        let mut user_turn = true; // user starts
        let mut next = 0;        // will store the next token in the sequence
        let mut pos = 0;     // position in the sequence
        while pos < steps {

            // when it is the user's turn to contribute tokens to the dialog...
            if user_turn {
                // get the (optional) system prompt at position 0
                let system_prompt = if pos == 0 {
                    // at position 0, the user can also contribute a system prompt
                    match cli_system_prompt {
                        None => {
                            // system prompt was not passed in, attempt to get it from stdin
                            // read_stdin("Enter system prompt (optional): ", system_prompt, sizeof(system_prompt));
                            utilities::read_stdin("Enter system prompt (optional): ")?
                        }
                        Some(cli_system_prompt) => {
                            // system prompt was passed in, use it
                            cli_system_prompt.clone()
                        }
                    }
                } else {
                    String::new()
                };
                // get the user prompt
                let user_prompt = if pos == 0 && !cli_user_prompt.is_empty() {
                    // user prompt for position 0 was passed in, use it
                    cli_user_prompt.clone()
                } else {
                    // otherwise get user prompt from stdin
                    utilities::read_stdin("User: ")?
                };
                // render user/system prompts into the Llama 2 Chat schema
                let rendered_prompt = if pos == 0 && !system_prompt.is_empty() {
                    format!("[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST]")
                } else {
                    format!("[INST] {user_prompt} [/INST]")
                };
                prompt_tokens = tokenizer.encode(&rendered_prompt, true, false)?;
                user_idx = 0; // reset the user index
                user_turn = false;
                print!("Assistant: ");
            }
            // token: the current token to feed into the transformer
            let token = if user_idx < prompt_tokens.len() {
                // if we are still processing the input prompt, force the next prompt token
                let token = prompt_tokens[user_idx];
                user_idx += 1;
                token
            } else {
                // otherwise use the next token sampled from previous turn
                next
            };
            // EOS (=2) token ends the Assistant turn
            if token == 2 { user_turn = true; }

            // forward the transformer to get logits for the next token
            runstate.forward(self, token, pos)?;
            next = sampler.sample(&mut runstate.logits) as usize;
            pos += 1;

            if user_idx >= prompt_tokens.len() && next != 2 {
                // the Assistant is responding, so print its output
                let piece = tokenizer.decode(token as i32, next as i32);
                print!("{}", piece);
            }
            if next == 2 { println!(); }
        }
        println!();
        Ok(())
    }
}
