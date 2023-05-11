use tract_onnx::{
    prelude::{DatumExt, Framework, InferenceModelExt},
    tract_hir::internal::Factoid,
};

fn main() {
    env_logger::init();
    let reader = &mut std::fs::File::open("./src/network.onnx").unwrap();
    let mut model = tract_onnx::onnx().model_for_read(reader).unwrap();

    for (i, id) in model.clone().inputs.iter().enumerate() {
        let input = model.node(id.node);

        let mut dims = vec![];
        let extracted_dims: Vec<usize> = input.outputs[0]
            .fact
            .shape
            .dims()
            .filter_map(|x| x.concretize())
            .map(|x| match x.to_i64() {
                Ok(x) => x as usize,
                Err(_e) => {
                    if x.to_string() == "batch_size" {
                        1
                    } else {
                        panic!("Unknown dimension {}: {:?}", x.to_string(), x)
                    }
                }
            })
            .collect();

        dims.extend(extracted_dims);

        model = model.with_input_fact(i, f32::fact(dims).into()).unwrap();
    }

    for (i, id) in model.clone().outputs.iter().enumerate() {
        let output = model.node(id.node);

        // add batch dim
        let mut dims = vec![];
        let extracted_dims: Vec<usize> = output.outputs[0]
            .fact
            .shape
            .dims()
            .filter_map(|x| x.concretize())
            .map(|x| match x.to_i64() {
                Ok(x) => x as usize,
                Err(_e) => {
                    if x.to_string() == "batch_size" {
                        1
                    } else {
                        panic!("Unknown dimension: {}", x)
                    }
                }
            })
            .collect();
        dims.extend(extracted_dims);

        println!("dims: {:?}", dims);

        model = model.with_output_fact(i, f32::fact(dims).into()).unwrap();
    }
    // Note: do not optimize the model, as the layout will depend on underlying hardware
    model.into_typed().unwrap().into_decluttered().unwrap();
}
