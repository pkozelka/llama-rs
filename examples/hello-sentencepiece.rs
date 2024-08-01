use sentencepiece_model::SentencePieceModel;


// Note that sentencepiece_model requires linux package `libssl-dev` to be installed

fn main() -> anyhow::Result<()> {

    // let model = SentencePieceModel::from_file("../llama2.c/tokenizer.model")?;
    let model = SentencePieceModel::from_file("../../huggingface/Llama-2-7b/tokenizer.model")?;
    assert_eq!(model.pieces.len(), 32000, "pieces.len");
    let trainer = model.trainer().unwrap();
    assert_eq!(trainer.unk_id(), 0, "unk_id");
    assert_eq!(trainer.bos_id(), 1, "bos_id");
    assert_eq!(trainer.eos_id(), 2, "eos_id");
    assert_eq!(trainer.pad_id(), -1, "pad_id");


    println!("model.denormalizer_spec: {:?}", model.denormalizer_spec);
    println!("model.normalizer_spec: {:?}", model.normalizer_spec);
    println!("model.trainer_spec: {:?}", model.trainer_spec);
    println!("model.self_test_data: {:?}", model.self_test_data);



    // show all the pieces, one per line
    for (i, piece) in model.pieces.iter().enumerate() {
        println!("{i:5}. {:6.2} {}", piece.score.unwrap(), piece.piece.as_deref().unwrap());
        if i > 10 {
            break;
        }
    }
    Ok(())
}