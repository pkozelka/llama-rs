//! ----------------------------------------------------------------------------
//! The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

use std::io::Read;
use std::path::PathBuf;
use byteorder::{LittleEndian, ReadBytesExt};

pub struct Tokenizer {
    /// the vocabulary of the tokenizer
    vocab: Vec<String>,
    /// the scores of the vocabulary
    vocab_scores: Vec<f32>,
    /// the sorted vocabulary
    sorted_vocab: Vec<TokenIndex>,
    /// the maximum token length
    max_token_length: usize,
    /// the byte pieces
    byte_pieces: Vec<u8>,
}

struct TokenIndex {
    str: String,
    id: usize,
}

impl Tokenizer {

    pub fn build_tokenizer(tokenizer_path: &PathBuf, vocab_size: usize) -> anyhow::Result<Self> {
        // log::debug!("build_tokenizer(tokenizer_path='{}', vocab_size={})", tokenizer_path.display(), vocab_size);
        // read in the tokenizer file
        let mut tokenizer = Tokenizer {
            vocab: Vec::with_capacity(vocab_size),
            vocab_scores: Vec::with_capacity(vocab_size),
            sorted_vocab: Vec::new(),
            max_token_length: 0,
            byte_pieces: vec![0; 256 * 2],
        };

        // initialize the byte pieces
        for i in 0..256 {
            tokenizer.byte_pieces[i * 2] = i as u8;
            tokenizer.byte_pieces[i * 2 + 1] = 0;
        }

        // read in the file
        let file = std::fs::File::open(tokenizer_path)?;
        let mut reader = std::io::BufReader::new(file);

        // read the max token length
        tokenizer.max_token_length = reader.read_i32::<LittleEndian>()? as usize;

        // read in the vocab
        for _ in 0..vocab_size {
            let score = reader.read_f32::<LittleEndian>()?;
            let len = reader.read_u32::<LittleEndian>()?;
            let mut vocab = vec![0; len as usize];
            reader.read_exact(&mut vocab)?;
            tokenizer.vocab.push(String::from_utf8(vocab)?);
            tokenizer.vocab_scores.push(score);
        }
        Ok(tokenizer)
    }

    pub fn encode(&mut self, text: &str, bos: bool, eos: bool) -> anyhow::Result<Vec<usize>> {
        eprintln!("encode(text='{text}',\n  bos={bos}, eos={eos})");
        // encode the string text (input) into an upper-bound preallocated tokens[] array
        // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
        if text.is_empty() {
            panic!("cannot encode NULL text");
        }

        // lazily malloc and sort the vocabulary
        if self.sorted_vocab.is_empty() {
            let mut sorted_vocab = Vec::with_capacity(self.vocab.len());
            for id in 0..self.vocab.len() {
                let token_index = TokenIndex { str: self.vocab[id].clone(), id };
                // log::debug!("encode: vocab[{}]='{}'", id, self.vocab[id]);

                sorted_vocab.push(token_index);
            }
            sorted_vocab.sort_by(|a, b| a.str.cmp(&b.str));
            self.sorted_vocab = sorted_vocab;
        }

        let mut tokens = self.process_unicode_text(text, bos);
        tokens.iter().enumerate().for_each(|(i, t)| eprintln!("tokens[{i}]={t} '{}'", self.vocab[*t]));

        // merge the best consecutive pair each iteration, according the scores in vocab_scores
        loop {
            let mut best_score = -1e10;
            let mut best_id = usize::MAX;
            let mut best_idx: Option<usize> = None;

            for i in 0..tokens.len() - 1 {
                // check if we can merge the pair (tokens[i], tokens[i+1])
                let str_buffer = format!("{}{}", self.vocab[tokens[i] as usize], self.vocab[tokens[i + 1] as usize]);
                if let Some(id) = self.str_lookup(&str_buffer) {
                    if self.vocab_scores[id] > best_score {
                        // this merge pair exists in vocab! record its score and position
                        best_score = self.vocab_scores[id];
                        best_id = id;
                        best_idx = Some(i);
                    }
                }
            }

            if best_idx.is_none() {
                break; // we couldn't find any more pairs to merge, so we're done
            }

            let best_idx = match best_idx {
                Some(idx) => idx,
                None => break,
            };

            // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
            tokens[best_idx] = best_id;
            // delete token at position best_idx+1, shift the entire sequence back 1
            eprintln!("removing token: {}", tokens[best_idx + 1]);
            tokens.remove(best_idx + 1);
        }

        // add optional EOS (=2) token, if desired
        if eos {
            tokens.push(2);
        }
        // log::debug!("encode: n_tokens={}", tokens.len());

        Ok(tokens)
    }

    fn process_unicode_text(&self, text: &str, bos: bool) -> Vec<usize> {
        // start at 0 tokens
        let mut tokens = Vec::with_capacity(text.len());

        // add optional BOS (=1) token, if desired
        if bos {
            tokens.push(1);
        }

        // add_dummy_prefix is true by default
        // so prepend a dummy prefix token to the input string, but only if text != ""
        if !text.is_empty() {
            if let Some(dummy_prefix) = self.str_lookup(" ") {
                tokens.push(dummy_prefix)
            };
        }

        let mut str_buffer = String::with_capacity(10);
        for c in text.chars() {
            str_buffer.clear();
            str_buffer.push(c);
            match self.str_lookup(&str_buffer) {
                Some(id) => {
                    // log::debug!("encode: str_buffer: '{}' -> id={}", str_buffer, id);
                    tokens.push(id)
                },
                None => {
                    // PK: this is nearly impossible to happen
                    // log::debug!("encode: str_buffer: '{}' -> id=NONE", str_buffer);
                    // byte_fallback encoding: just encode each byte as a token
                    for byte in str_buffer.bytes() {
                        // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
                        // so the individual bytes only start at index 3
                        let id = byte as usize + 3;
                        tokens.push(id);
                    }
                }
            }
        }
        //
        tokens
    }

    fn str_lookup(&self, str: &str) -> Option<usize> {
        // efficiently find the perfect match for str in vocab, return its index or -1 if not found
        let tok = TokenIndex { str: str.to_string(), id: 0 }; // acts as the key to search for
        match self.sorted_vocab.binary_search_by(|a| a.str.cmp(&tok.str)) {
            Ok(idx) => Some(self.sorted_vocab[idx].id),
            Err(_) => None,
        }
    }

    pub fn decode(&self, prev_token: i32, token: i32) -> String {
        let piece = &self.vocab[token as usize];
        // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
        if prev_token == 1 && piece.starts_with(' ') {
            return piece[1..].to_string();
        }
        // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
        // parse this and convert and return the actual byte
        if let Some(byte_val) = piece.strip_prefix("<0x") {
            if let Ok(byte_val) = u8::from_str_radix(byte_val, 16) {
                let byte_val = byte_val as usize;
                let byte_val = byte_val * 2;
                let byte_val = byte_val as usize;
                let byte_val = byte_val as usize;
                return String::from_utf8_lossy(&self.byte_pieces[byte_val..byte_val + 2]).to_string();
            }
        }
        piece.to_string()
    }

}
