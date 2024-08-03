use crate::llama2::sampler::Sampler;
use crate::llama2::tokenizer::Tokenizer;
use crate::llama2::{RunState, Transformer, utilities};

impl Transformer {

    /// chat loop: comment from the original C code:
    /// > I manually inspected the tokens for a few chat conversations compared to
    /// > python reference and that seemed ok, but this was not thoroughly tested and
    /// > is not safely implemented, it's more a proof of concept atm.
    pub(crate) fn chat(&self, tokenizer: &Tokenizer, sampler: &Sampler, cli_user_prompt: &String, cli_system_prompt: &Option<String>, steps: usize) -> anyhow::Result<()> {
        let  mut runstate = RunState::malloc_run_state(&self.config);

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

