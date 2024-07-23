use crate::run::sampler::Sampler;
use crate::run::tokenizer::Tokenizer;
use crate::run::Transformer;

/* convert from C:

void chat(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler,
          char *cli_user_prompt, char *cli_system_prompt, int steps) {

    // buffers for reading the system prompt and user prompt from stdin
    // you'll notice they are soomewhat haphazardly and unsafely set atm
    char system_prompt[512];
    char user_prompt[512];
    char rendered_prompt[1152];
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc(1152 * sizeof(int));
    int user_idx;

    // start the main loop
    int8_t user_turn = 1; // user starts
    int next;        // will store the next token in the sequence
    int token;       // stores the current token to feed into the transformer
    int prev_token;
    int pos = 0;     // position in the sequence
    while (pos < steps) {

        // when it is the user's turn to contribute tokens to the dialog...
        if (user_turn) {
            // get the (optional) system prompt at position 0
            if (pos == 0) {
                // at position 0, the user can also contribute a system prompt
                if (cli_system_prompt == NULL) {
                    // system prompt was not passed in, attempt to get it from stdin
                    read_stdin("Enter system prompt (optional): ", system_prompt, sizeof(system_prompt));
                } else {
                    // system prompt was passed in, use it
                    strcpy(system_prompt, cli_system_prompt);
                }
            }
            // get the user prompt
            if (pos == 0 && cli_user_prompt != NULL) {
                // user prompt for position 0 was passed in, use it
                strcpy(user_prompt, cli_user_prompt);
            } else {
                // otherwise get user prompt from stdin
                read_stdin("User: ", user_prompt, sizeof(user_prompt));
            }
            // render user/system prompts into the Llama 2 Chat schema
            if (pos == 0 && system_prompt[0] != '\0') {
                char system_template[] = "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]";
                sprintf(rendered_prompt, system_template, system_prompt, user_prompt);
            } else {
                char user_template[] = "[INST] %s [/INST]";
                sprintf(rendered_prompt, user_template, user_prompt);
            }
            // encode the rendered prompt into tokens
            encode(tokenizer, rendered_prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
            user_idx = 0; // reset the user index
            user_turn = 0;
            printf("Assistant: ");
        }

        // determine the token to pass into the transformer next
        if (user_idx < num_prompt_tokens) {
            // if we are still processing the input prompt, force the next prompt token
            token = prompt_tokens[user_idx++];
        } else {
            // otherwise use the next token sampled from previous turn
            token = next;
        }
        // EOS (=2) token ends the Assistant turn
        if (token == 2) { user_turn = 1; }

        // forward the transformer to get logits for the next token
        float* logits = forward(transformer, token, pos);
        next = sample(sampler, logits);
        pos++;

        if (user_idx >= num_prompt_tokens && next != 2) {
            // the Assistant is responding, so print its output
            char* piece = decode(tokenizer, token, next);
            safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
            fflush(stdout);
        }
        if (next == 2) { printf("\n"); }
    }
    printf("\n");
    free(prompt_tokens);
}



 */
impl Transformer {

    /// chat loop: comment from the original C code:
    /// > I manually inspected the tokens for a few chat conversations compared to
    /// > python reference and that seemed ok, but this was not thoroughly tested and
    /// > is not safely implemented, it's more a proof of concept atm.
    pub(crate) fn chat(&mut self, tokenizer: &mut Tokenizer, sampler: &mut Sampler, cli_user_prompt: &String, cli_system_prompt: &Option<String>, steps: usize) -> anyhow::Result<()> {
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
                            read_stdin("Enter system prompt (optional): ")?
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
                    read_stdin("User: ")?
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
            self.forward(token, pos)?;
            next = sampler.sample(&mut self.state.logits) as usize;
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

fn read_stdin(message: &str) -> anyhow::Result<String> {
    println!("{message}");
    let mut buf = String::new();
    match std::io::stdin().read_line(&mut buf) {
        Ok(_) => Ok(buf),
        Err(e) => Err(anyhow::anyhow!("error reading stdin: {e}"))
    }
}
