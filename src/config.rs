use std::io::{Read, Seek};
use byteorder::{LittleEndian, ReadBytesExt};

/// Transformer model
#[derive(Default, Debug)]
pub struct Config {
    /// transformer dimension
    pub dim: usize,
    /// for ffn layers
    pub hidden_dim: usize,
    /// number of layers
    pub n_layers: usize,
    /// number of query heads
    pub n_heads: usize,
    /// number of key/value heads (can be < query heads because of multiquery)
    pub n_kv_heads: usize,
    /// vocabulary size, usually 256 (byte-level)
    pub vocab_size: usize,
    /// max sequence length
    pub seq_len: usize,
    /// `true` indicates that we can reuse `model.tok_embeddings.weight` for `model.output.weight`
    pub shared_weights: bool,
    /// group size used for quantization
    pub group_size: Option::<usize>,
}

/// Model format
#[derive(Debug, PartialEq)]
pub enum ModelFormat {
    Legacy,
    V1,
    V2,
    //TODO? HuggingFace,
}

impl Config {

    /// Read the model configuration from a reader and position the reader at the start of the model data.
    pub fn read_config<R: Read + Seek>(reader: &mut R) -> anyhow::Result<(Self, ModelFormat)> {
        let pos = reader.stream_position()?;
        let magic = reader.read_u32::<LittleEndian>()?;
        if magic == 0x616b3432/*b"ak42"*/ {
            let version = reader.read_i32::<LittleEndian>()?;
            let mut config = Self::read_config_legacy(reader)?;
            let shared_weights = reader.read_u8()?;
            // C99 _Bool representation:
            config.shared_weights = shared_weights != 0;
            let model_format = match version {
                1 => ModelFormat::V1,
                2 => {
                    let group_size = reader.read_i32::<LittleEndian>()?;
                    log::debug!("group_size: {}", group_size);
                    config.group_size = Some(group_size as usize);
                    ModelFormat::V2
                }
                _ => return Err(anyhow::anyhow!("unsupported model version: {}", version)),
            };
            Ok((config, model_format))
        } else {
            reader.seek(std::io::SeekFrom::Start(pos))?;
            let config = Self::read_config_legacy(reader)?;
            Ok((config, ModelFormat::Legacy))
        }
    }
    fn read_config_legacy<R: Read + Seek>(reader: &mut R) -> anyhow::Result<Self> {
        let dim = reader.read_u32::<LittleEndian>()?;
        let hidden_dim = reader.read_u32::<LittleEndian>()?;
        let n_layers = reader.read_u32::<LittleEndian>()?;
        let n_heads = reader.read_u32::<LittleEndian>()?;
        let n_kv_heads = reader.read_u32::<LittleEndian>()?;
        let vocab_size = reader.read_i32::<LittleEndian>()?;
        let seq_len = reader.read_u32::<LittleEndian>()?;

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
            group_size: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use super::*;

    #[ignore]
    #[test]
    fn legacy_stories15m_bin() -> anyhow::Result<()>{
        let mut file = File::open("downloads/stories15M.bin")?;
        let (config, model_format) = Config::read_config(&mut file).unwrap();
        assert_eq!(model_format, ModelFormat::Legacy, "Legacy model format expected");
        assert_eq!(config.dim, 288, "dim");
        assert_eq!(config.hidden_dim, 768, "hidden_dim");
        assert_eq!(config.n_layers, 6, "n_layers");
        assert_eq!(config.n_heads, 6, "n_heads");
        assert_eq!(config.n_kv_heads, 6, "n_kv_heads");
        assert_eq!(config.vocab_size, 32000, "vocab_size");
        assert_eq!(config.seq_len, 256, "seq_len");
        assert_eq!(config.shared_weights, true, "shared_weights");
        assert_eq!(config.group_size, None, "group_size");
        Ok(())
    }

    #[ignore]
    #[test]
    fn legacy_stories42m_bin()  -> anyhow::Result<()>{
        let mut file = File::open("downloads/stories42M.bin")?;
        let (config, model_format) = Config::read_config(&mut file).unwrap();
        assert_eq!(model_format, ModelFormat::Legacy, "Legacy model format expected");
        assert_eq!(config.dim, 512, "dim");
        assert_eq!(config.hidden_dim, 1376, "hidden_dim");
        assert_eq!(config.n_layers, 8, "n_layers");
        assert_eq!(config.n_heads, 8, "n_heads");
        assert_eq!(config.n_kv_heads, 8, "n_kv_heads");
        assert_eq!(config.vocab_size, 32000, "vocab_size");
        assert_eq!(config.seq_len, 1024, "seq_len");
        assert_eq!(config.shared_weights, true, "shared_weights");
        assert_eq!(config.group_size, None, "group_size");
        Ok(())
    }

    #[ignore]
    #[test]
    fn llama2_7b_q80_bin()  -> anyhow::Result<()>{
        let mut file = File::open("downloads/llama2_7b_q80.bin")?;
        let (config, model_format) = Config::read_config(&mut file).unwrap();
        assert_eq!(model_format, ModelFormat::V2, "model format");
        assert_eq!(config.dim, 4096, "dim");
        assert_eq!(config.hidden_dim, 11008, "hidden_dim");
        assert_eq!(config.n_layers, 32, "n_layers");
        assert_eq!(config.n_heads, 32, "n_heads");
        assert_eq!(config.n_kv_heads, 32, "n_kv_heads");
        assert_eq!(config.vocab_size, 32000, "vocab_size");
        assert_eq!(config.seq_len, 2048, "seq_len");
        assert_eq!(config.shared_weights, false, "shared_weights");
        assert_eq!(config.group_size, Some(64), "group_size");
        Ok(())
    }

    #[ignore ="crashes due to unknown model"]
    #[test]
    fn llama3_8b_instruct_bin()  -> anyhow::Result<()>{
        let mut file = File::open("downloads/llama3_8b_instruct.bin")?;
        let (config, model_format) = Config::read_config(&mut file).unwrap();
        assert_eq!(model_format, ModelFormat::Legacy, "model format");
        assert_eq!(config.dim, 4096, "dim");
        assert_eq!(config.hidden_dim, 14336, "hidden_dim");
        assert_eq!(config.n_layers, 32, "n_layers");
        assert_eq!(config.n_heads, 32, "n_heads");
        assert_eq!(config.n_kv_heads, 8, "n_kv_heads");
        assert_eq!(config.vocab_size, 128256, "vocab_size");
        assert_eq!(config.seq_len, 2048, "seq_len");
        assert_eq!(config.shared_weights, false, "shared_weights");
        assert_eq!(config.group_size, Some(64), "group_size");
        Ok(())
    }
}