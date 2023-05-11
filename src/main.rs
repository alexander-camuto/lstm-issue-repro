use tract_onnx::prelude::{Framework, InferenceFact, InferenceModelExt, SymbolValues};

fn main() {
    env_logger::init();
    let reader = &mut std::fs::File::open("./src/network.onnx").unwrap();
    let mut model = tract_onnx::onnx().model_for_read(reader).unwrap();

    for (i, _) in model.clone().outputs.iter().enumerate() {
        model.set_output_fact(i, InferenceFact::default()).unwrap();
    }
    // Note: do not optimize the model, as the layout will depend on underlying hardware
    let model = model.into_typed().unwrap().into_decluttered().unwrap();
    let batch_size = model.symbol_table.sym("batch_size");
    let seq_len = model.symbol_table.sym("sequence_length");
    model
        .concretize_dims(&SymbolValues::default().with(&batch_size, 1))
        .unwrap()
        .concretize_dims(&SymbolValues::default().with(&seq_len, 1))
        .unwrap();
    // Note: do not optimize the model, as the layout will depend on underlying hardware
}
