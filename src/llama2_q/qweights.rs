use std::fs::File;
use std::io::BufReader;

use llama_rs::config::Config;
use llama_rs::utilities;

use crate::llama2_q::q80::QuantizedTensor;

#[derive(Default)]
pub struct QTransformerWeights {
    // token embedding table
    /// (vocab_size, dim)
    pub(crate) q_tokens: QuantizedTensor,
    /// same, but dequantized
    pub(crate) token_embedding_table: Vec<f32>,

    // weights for rmsnorms
    /// (layer, dim) rmsnorm weights
    pub(crate) rms_att_weight: Vec<f32>,
    /// (layer, dim)
    pub(crate) rms_ffn_weight: Vec<f32>,
    /// (layer, dim, n_heads * head_size)
    pub(crate) wq: Vec<QuantizedTensor>,
    /// (layer, dim, n_kv_heads * head_size)
    pub(crate) wk: Vec<QuantizedTensor>,
    /// (layer, dim, n_kv_heads * head_size)
    pub(crate) wv: Vec<QuantizedTensor>,
    /// (layer, n_heads * head_size, dim)
    pub(crate) wo: Vec<QuantizedTensor>,

    // weights for ffn
    /// (layer, hidden_dim, dim)
    pub(crate) w1: Vec<QuantizedTensor>,
    /// (layer, dim, hidden_dim)
    pub(crate) w2: Vec<QuantizedTensor>,
    /// (layer, hidden_dim, dim)
    pub(crate) w3: Vec<QuantizedTensor>,

    // final rmsnorm
    /// (dim,)
    pub(crate) rms_final_weight: Vec<f32>,
    /// (optional) classifier weights for the logits, on the last layer
    pub(crate) wcls: QuantizedTensor,
}

impl QTransformerWeights {
    /// currently we have this instead of memory_map_weights()
    pub(crate) fn read_weights(reader: &mut BufReader<File>, p: &Config) -> anyhow::Result<Self> {
        let head_size = p.dim / p.n_heads;
        let hidden_dim = p.hidden_dim;
        let n_layers = p.n_layers;

        let rms_att_weight = utilities::read_f32_table(reader, n_layers, p.dim)?;
        let rms_ffn_weight = utilities::read_f32_table(reader, n_layers, p.dim)?;
        let rms_final_weight = utilities::read_f32_table(reader, 1, p.dim)?;

        let q_tokens = QuantizedTensor::init_quantized_tensors(reader, 1, p.vocab_size * p.dim)?
            .into_iter()
            .next()
            .unwrap();

        let token_embedding_table = q_tokens.dequantize();

        let wq = QuantizedTensor::init_quantized_tensors(reader, n_layers, p.dim * (p.n_heads * head_size))?;
        let wk = QuantizedTensor::init_quantized_tensors(reader, n_layers, p.dim * (p.n_kv_heads * head_size))?;
        let wv = QuantizedTensor::init_quantized_tensors(reader, n_layers, p.dim * (p.n_kv_heads * head_size))?;
        let wo = QuantizedTensor::init_quantized_tensors(reader, n_layers, p.dim * (p.n_heads * head_size))?;
        //

        let w1 = QuantizedTensor::init_quantized_tensors(reader, n_layers, p.dim * hidden_dim)?;
        let w2 = QuantizedTensor::init_quantized_tensors(reader, n_layers, p.dim * hidden_dim)?;
        let w3 = QuantizedTensor::init_quantized_tensors(reader, n_layers, p.dim * hidden_dim)?;


        let wcls = if p.shared_classifier {
            q_tokens.clone()
        } else {
            QuantizedTensor::init_quantized_tensors(reader, 1, p.vocab_size * p.dim)?
                .into_iter()
                .next()
                .unwrap()
        };

        Ok(Self {
            q_tokens,
            token_embedding_table,
            rms_att_weight,
            wq,
            wk,
            wv,
            wo,
            rms_ffn_weight,
            w1,
            w2,
            w3,
            rms_final_weight,
            wcls,
        })
    }

}