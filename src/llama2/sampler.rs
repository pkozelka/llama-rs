//! ----------------------------------------------------------------------------
//! The Sampler, which takes logits and returns a sampled token
//! sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

use crate::llama2::math::softmax;

#[derive(Debug)]
pub struct ProbIndex {
    prob: f32,
    index: i32,
}

pub struct Sampler {
    /// the vocabulary size
    vocab_size: usize,
    /// buffer used in top-p sampling
    _probindex: Vec<ProbIndex>,
    /// the temperature of the sampling
    temperature: f32,
    /// the top-p value
    topp: f32,
    /// the random number generator seed
    rng_seed: u64,
}

impl Sampler {
    pub(crate) fn build_sampler(vocab_size: usize, temperature: f32, topp: f32, rng_seed: u64) -> anyhow::Result<Self> {
        // buffer only used with nucleus sampling; may not need but it's ~small
        let probindex = Vec::with_capacity(vocab_size);
        //
        Ok(Self { vocab_size, _probindex: probindex, temperature, topp, rng_seed })
    }

    pub fn sample(&mut self, logits: &mut [f32]) -> i32 {
        // eprintln!("temperature: {}, topp: {}", self.temperature, self.topp);
        // print first 20 logits
        // eprintln!("sample::logits.len={}", logits.len());
        logits.iter().take(20).enumerate().for_each(|(i,l)| eprintln!("logit[{i}]={l:.6}"));

        // sample the token given the logits and some hyperparameters
        if self.temperature == 0.0 {
            // greedy argmax sampling: take the token with the highest probability
            sample_argmax(logits)
        } else {
            // apply the temperature to the logits
            for q in 0..self.vocab_size {
                logits[q] /= self.temperature;
            }
            // apply softmax to the logits to get the probabilities for next token
            softmax(logits);
            // flip a (float) coin (this is our source of entropy for sampling)
            let coin = random_f32(&mut self.rng_seed);
            // eprintln!("sample::coin: {:.6}", coin);
            // we sample from this distribution to get the next token
            if self.topp <= 0.0 || self.topp >= 1.0 {
                // simply sample from the predicted probability distribution
                sample_mult(&logits, coin)
            } else {
                // top-p (nucleus) sampling, clamping the least likely tokens to zero
                sample_topp(&logits, self.topp, coin)
            }
        }
    }
}

fn random_u32(state: &mut u64) -> u32 {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state = *state ^ (*state >> 12);
    *state ^= *state << 25;
    *state ^= *state >> 27;
    (state.wrapping_mul(0x2545F4914F6CDD1D_u64) >> 32) as u32
}

fn random_f32(state: &mut u64) -> f32 {
    // random float32 in [0,1)
    (random_u32(state) >> 8) as f32 / 16777216.0
}

fn sample_topp(probabilities: &[f32], topp: f32, /*probindex: &[ProbIndex], */coin: f32) -> i32 {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".
    // coin is a random number in [0, 1), usually from random_f32()

    // quicksort indices in descending order of probabilities
    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before sorting
    let cutoff = (1.0 - topp) / (probabilities.len() as f32 - 1.0);
    let mut probindex = Vec::with_capacity(probabilities.len());
    for i in 0..probabilities.len() {
        if probabilities[i] >= cutoff {
            probindex.push(ProbIndex { index: i as i32, prob: probabilities[i] });
        }
    }
    probindex.sort_by(|a, b| b.prob.partial_cmp(&a.prob).unwrap());

    // log first 10 of probindex
    eprintln!("sample_topp::probindex.len={} probabilities.len={} cutoff={cutoff:.6}", probindex.len(), probabilities.len());
    // probindex.iter().take(10).for_each(|pi| eprintln!("* {pi:?}"));


    // truncate the list where cumulative probability exceeds topp
    let mut cumulative_prob = 0.0;
    let mut last_idx = probindex.len() - 1; // in case of rounding errors consider all elements
    for i in 0..probindex.len() {
        cumulative_prob += probindex[i].prob;
        if cumulative_prob > topp {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }

    // sample from the truncated list
    let r = coin * cumulative_prob;
    let mut cdf = 0.0;
    for i in 0..=last_idx {
        cdf += probindex[i].prob;
        if r < cdf {
            return probindex[i].index;
        }
    }
    probindex[last_idx as usize].index // in case of rounding errors
}



fn sample_argmax(probabilities: &[f32]) -> i32 {
    // return the index that has the highest probability
    let mut max_i = 0;
    let mut max_p = probabilities[0];
    for (i, &p) in probabilities.iter().enumerate() {
        if p > max_p {
            max_i = i;
            max_p = p;
        }
    }
    max_i as i32
}

fn sample_mult(probabilities: &[f32], coin: f32) -> i32 {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    let mut cdf = 0.0;
    for (i, &p) in probabilities.iter().enumerate() {
        cdf += p;
        if coin < cdf {
            return i as i32;
        }
    }
    probabilities.len() as i32 - 1 // in case of rounding errors
}
