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
    /// the size of the vocabulary
    vocab_size: usize,
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

    /* from C:
void build_tokenizer(Tokenizer* t, char* tokenizer_path, int vocab_size) {
    // i should have written the vocab_size into the tokenizer file... sigh
    t->vocab_size = vocab_size;
    // malloc space to hold the scores and the strings
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL; // initialized lazily
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    // read in the file
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) { fprintf(stderr, "couldn't load %s\n", tokenizer_path); exit(EXIT_FAILURE); }
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE);}
        if (fread(&len, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i] = (char *)malloc(len + 1);
        if (fread(t->vocab[i], len, 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i][len] = '\0'; // add the string terminating token
    }
    fclose(file);
}

     */
    pub(crate) fn build_tokenizer(tokenizer_path: &PathBuf, vocab_size: usize) -> anyhow::Result<Self> {
        // read in the tokenizer file
        let mut tokenizer = Tokenizer {
            vocab: Vec::with_capacity(vocab_size),
            vocab_scores: Vec::with_capacity(vocab_size),
            sorted_vocab: Vec::new(),
            vocab_size,
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
        reader.read_i32::<LittleEndian>()?;

        // read in the vocab
        for _ in 0..vocab_size {
            let score = reader.read_f32::<LittleEndian>()?;
            let len = reader.read_i32::<LittleEndian>()?;
            let mut vocab = vec![0; len as usize];
            reader.read_exact(&mut vocab)?;
            tokenizer.vocab.push(String::from_utf8(vocab)?);
            tokenizer.vocab_scores.push(score);
        }
        Ok(tokenizer)
    }

    pub fn encode(&mut self, text: &str, bos: bool, eos: i8) -> anyhow::Result<Vec<i32>> {
        // encode the string text (input) into an upper-bound preallocated tokens[] array
        // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
        if text.is_empty() {
            panic!("cannot encode NULL text");
        }

        // lazily malloc and sort the vocabulary
        if self.sorted_vocab.is_empty() {
            let mut sorted_vocab = Vec::with_capacity(self.vocab_size);
            for i in 0..self.vocab_size {
                sorted_vocab.push(TokenIndex { str: self.vocab[i].clone(), id: i });
            }
            sorted_vocab.sort_by(|a, b| a.str.cmp(&b.str));
            self.sorted_vocab = sorted_vocab;
        }

        let mut tokens = self.process_text(text, bos);

        // merge the best consecutive pair each iteration, according the scores in vocab_scores
        loop {
            let mut best_score = -1e10;
            let mut best_id = -1;
            let mut best_idx: Option<usize> = None;

            for i in 0..tokens.len() - 1 {
                // check if we can merge the pair (tokens[i], tokens[i+1])
                let str_buffer = format!("{}{}", self.vocab[tokens[i] as usize], self.vocab[tokens[i + 1] as usize]);
                let id = self.str_lookup(&str_buffer);
                if id != -1 && self.vocab_scores[id] > best_score {
                    // this merge pair exists in vocab! record its score and position
                    best_score = self.vocab_scores[id];
                    best_id = id;
                    best_idx = Some(i);
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
            tokens.remove(best_idx + 1);
        }

        // add optional EOS (=2) token, if desired
        if eos != 0 {
            tokens.push(2);
        }

        Ok(tokens)
    }

    fn process_text(&self, text: &str, bos: bool) -> Vec<i32> {
        // create a temporary buffer that will store merge candidates of always two consecutive tokens
        // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
        let mut str_buffer = vec![0; self.max_token_length * 2 + 1 + 2];
        let mut str_len = 0;

        // start at 0 tokens
        let mut tokens = Vec::with_capacity(text.len());

        // add optional BOS (=1) token, if desired
        if bos {
            tokens.push(1);
        }

        // add_dummy_prefix is true by default
        // so prepend a dummy prefix token to the input string, but only if text != ""
        // TODO: pretty sure this isn't correct in the general case but I don't have the
        // energy to read more of the sentencepiece code to figure out what it's doing
        if !text.is_empty() {
            let dummy_prefix = self.str_lookup(" ");
            tokens.push(dummy_prefix);
        }

        // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
        // Code point ↔ UTF-8 conversion
        // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
        // U+0000	U+007F	    0xxxxxxx
        // U+0080	U+07FF	    110xxxxx	10xxxxxx
        // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
        // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

        // process the raw (UTF-8) byte sequence of the input string

        for c in text.bytes() {
            // reset buffer if the current byte is ASCII or a leading byte
            // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
            // 0x80 is 10000000
            // in UTF-8, all continuation bytes start with "10" in first two bits
            // so in English this is: "if this byte is not a continuation byte"
            if (c & 0xC0) != 0x80 {
                // this byte must be either a leading byte (11...) or an ASCII char (0x...)
                // => reset our location, as we're starting a new UTF-8 codepoint
                str_len = 0;
            }

            // append the current byte to the buffer
            str_buffer[str_len] = c;
            str_len += 1;
            str_buffer[str_len] = 0;

            // while the next character is a continuation byte, continue appending
            // but if there are too many of them, just stop to avoid overruning str_buffer size.
            if text.bytes().nth(1).unwrap() & 0xC0 == 0x80 && str_len < 4 {
                continue;
            }

            // ok c+1 is not a continuation byte, so we've read in a full codepoint
            let id = self.str_lookup(&str_buffer);

            if id != -1 {
                // we found this codepoint in vocab, add it as a token
                tokens.push(id);
            } else {
                // byte_fallback encoding: just encode each byte as a token
                // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
                // so the individual bytes only start at index 3
                for i in 0..str_len {
                    tokens.push(str_buffer[i] as i32 + 3);
                }
            }
            str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
        }
        tokens
    }

    fn str_lookup(&self, str: &str) -> i32 {
        // efficiently find the perfect match for str in vocab, return its index or -1 if not found
        let tok = TokenIndex { str: str.to_string(), id: 0 }; // acts as the key to search for
        match self.sorted_vocab.binary_search_by(|a| a.str.cmp(&tok.str)) {
            Ok(idx) => self.sorted_vocab[idx].id as i32,
            Err(_) => -1,
        }
    }
}
