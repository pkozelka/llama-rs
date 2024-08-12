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
    /// legacy format only - derived from negative vocab_size
    pub shared_weights: bool,
    /// a byte to indicate if the classifier is shared
    /// non legacy format only
    pub shared_classifier: bool,
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
    const AK42_MAGIC: u32 = 0x616b3432; // b"ak42"

    /// Read the model configuration from a reader and position the reader at the start of the model data.
    pub fn read_config<R: Read + Seek>(reader: &mut R) -> anyhow::Result<Self> {
        let pos = reader.stream_position()?;
        let magic = reader.read_u32::<LittleEndian>()?;
        if magic == Self::AK42_MAGIC {
            let version = reader.read_i32::<LittleEndian>()?;
            log::debug!("AK42 model format version {version}");
            match version {
                1 => {},
                2 => {}
                _ => return Err(anyhow::anyhow!("unsupported AK42 model version: {}", version)),
            };
            let config = Self::read_config_params(reader, version)?;
            reader.seek(std::io::SeekFrom::Start(pos + 256))?;
            Ok(config)
        } else {
            // legacy version, without magic and explicit version specified
            log::debug!("legacy model format");
            reader.seek(std::io::SeekFrom::Start(pos))?;
            let config = Self::read_config_params(reader, 0)?;
            Ok(config)
        }
    }
    fn read_config_params<R: Read + Seek>(reader: &mut R, version: i32) -> anyhow::Result<Self> {
        let dim = reader.read_u32::<LittleEndian>()?;
        let hidden_dim = reader.read_u32::<LittleEndian>()?;
        let n_layers = reader.read_u32::<LittleEndian>()?;
        let n_heads = reader.read_u32::<LittleEndian>()?;
        let n_kv_heads = reader.read_u32::<LittleEndian>()?;
        let vocab_size_i32 = reader.read_i32::<LittleEndian>()?;
        let seq_len = reader.read_u32::<LittleEndian>()?;

        let shared_weights;
        let shared_classifier;
        let vocab_size;
        let group_size;
        if version == 0 {
            // negative vocab size is hacky way of signaling unshared weights. bit yikes.
            shared_weights = vocab_size_i32 > 0;
            shared_classifier = false;
            vocab_size = vocab_size_i32.abs() as usize;
            group_size = None;
        } else {
            shared_weights = false;
            shared_classifier = reader.read_i8()? > 0;
            vocab_size = vocab_size_i32 as usize;
            group_size = if version == 1 {
                None
            } else {
                let gsz = reader.read_i32::<LittleEndian>()?;
                log::debug!("group_size: {}", gsz);
                Some(gsz as usize)
            };
        }

        Ok(Self {
            dim: dim as usize,
            hidden_dim: hidden_dim as usize,
            n_layers: n_layers as usize,
            n_heads: n_heads as usize,
            n_kv_heads: n_kv_heads as usize,
            vocab_size,
            seq_len: seq_len as usize,
            shared_weights,
            shared_classifier,
            group_size,
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
        env_logger::builder().filter_level(log::LevelFilter::Debug).init();

        let mut file = File::open("downloads/stories15M.bin")?;
        let config = Config::read_config(&mut file).unwrap();
        println!("{config:#?}");

        assert_eq!(config.dim, 288, "dim");
        assert_eq!(config.hidden_dim, 768, "hidden_dim");
        assert_eq!(config.n_layers, 6, "n_layers");
        assert_eq!(config.n_heads, 6, "n_heads");
        assert_eq!(config.n_kv_heads, 6, "n_kv_heads");
        assert_eq!(config.vocab_size, 32000, "vocab_size");
        assert_eq!(config.seq_len, 256, "seq_len");
        assert_eq!(config.shared_weights, true, "shared_weights");
        assert_eq!(config.group_size, None, "group_size");
        assert_eq!(file.stream_position()?, 28, "stream_position");
        Ok(())
    }

    #[ignore]
    #[test]
    fn legacy_stories42m_bin()  -> anyhow::Result<()>{
        env_logger::builder().filter_level(log::LevelFilter::Debug).init();

        let mut file = File::open("downloads/stories42M.bin")?;
        let config = Config::read_config(&mut file).unwrap();
        println!("{config:#?}");

        assert_eq!(file.stream_position()?, 28, "stream_position");
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
        env_logger::builder().filter_level(log::LevelFilter::Trace).init();

        let mut file = File::open("downloads/llama2_7b_q80.bin")?;
        let config = Config::read_config(&mut file).unwrap();
        println!("{config:#?}");

        assert_eq!(file.stream_position()?, 256, "stream_position");
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

    #[ignore]
    #[test]
    fn llama3_8b_instruct_bin()  -> anyhow::Result<()>{
        env_logger::builder().filter_level(log::LevelFilter::Trace).init();

        let mut file = File::open("downloads/llama3_8b_instruct.bin")?;
        let config = Config::read_config(&mut file).unwrap();
        println!("{config:#?}");

        assert_eq!(config.dim, 4096, "dim");
        assert_eq!(config.hidden_dim, 14336, "hidden_dim");
        assert_eq!(config.n_layers, 32, "n_layers");
        assert_eq!(config.n_heads, 32, "n_heads");
        assert_eq!(config.n_kv_heads, 8, "n_kv_heads");
        assert_eq!(config.vocab_size, 128256, "vocab_size");
        assert_eq!(config.seq_len, 2048, "seq_len");
        assert_eq!(config.shared_weights, false, "shared_weights");
        assert_eq!(config.group_size, None, "group_size");
        Ok(())
    }
}