use std::cmp::max;

use ndarray::{Array1, ArrayD};
use onnxruntime::{environment::Environment, GraphOptimizationLevel};
use tokenizers::tokenizer::{Result, Tokenizer};
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

fn main() {
    // a builder for `FmtSubscriber`.
    let subscriber = FmtSubscriber::builder()
        // all spans/events with a level higher than TRACE (e.g, debug, info, warn, etc.)
        // will be written to stdout.
        .with_max_level(Level::INFO)
        // completes the builder.
        .finish();
    tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");
    let env = Environment::builder().with_name("env").build().unwrap();
    let mut session = env
        .new_session_builder()
        .unwrap()
        .with_optimization_level(GraphOptimizationLevel::All)
        .unwrap()
        .with_model_from_file("gpt2.onnx")
        .unwrap();

    // inputs:
    // Input { name: "input_ids", input_type: Int64, dimensions: [None, None] }
    // Input { name: "position_ids", input_type: Int64, dimensions: [None, None] }
    // Input { name: "attention_mask", input_type: Float, dimensions: [None, None] }
    // Input { name: "past_0", input_type: Float, dimensions: [Some(2), None, Some(12), None, Some(64)] }
    // Input { name: "past_1", input_type: Float, dimensions: [Some(2), None, Some(12), None, Some(64)] }
    // Input { name: "past_2", input_type: Float, dimensions: [Some(2), None, Some(12), None, Some(64)] }
    // Input { name: "past_3", input_type: Float, dimensions: [Some(2), None, Some(12), None, Some(64)] }
    // Input { name: "past_4", input_type: Float, dimensions: [Some(2), None, Some(12), None, Some(64)] }
    // Input { name: "past_5", input_type: Float, dimensions: [Some(2), None, Some(12), None, Some(64)] }
    // Input { name: "past_6", input_type: Float, dimensions: [Some(2), None, Some(12), None, Some(64)] }
    // Input { name: "past_7", input_type: Float, dimensions: [Some(2), None, Some(12), None, Some(64)] }
    // Input { name: "past_8", input_type: Float, dimensions: [Some(2), None, Some(12), None, Some(64)] }
    // Input { name: "past_9", input_type: Float, dimensions: [Some(2), None, Some(12), None, Some(64)] }
    // Input { name: "past_10", input_type: Float, dimensions: [Some(2), None, Some(12), None, Some(64)] }
    // Input { name: "past_11", input_type: Float, dimensions: [Some(2), None, Some(12), None, Some(64)]
    // outputs:
    // Output { name: "logits", output_type: Float, dimensions: [None, None, Some(50257)] }
    // Output { name: "present_0", output_type: Float, dimensions: [Some(2), None, Some(12), None, Some(64)] }
    // Output { name: "present_1", output_type: Float, dimensions: [Some(2), None, Some(12), None, Some(64)] }
    // Output { name: "present_2", output_type: Float, dimensions: [Some(2), None, Some(12), None, Some(64)] }
    // Output { name: "present_3", output_type: Float, dimensions: [Some(2), None, Some(12), None, Some(64)] }
    // Output { name: "present_4", output_type: Float, dimensions: [Some(2), None, Some(12), None, Some(64)] }
    // Output { name: "present_5", output_type: Float, dimensions: [Some(2), None, Some(12), None, Some(64)] }
    // Output { name: "present_6", output_type: Float, dimensions: [Some(2), None, Some(12), None, Some(64)] }
    // Output { name: "present_7", output_type: Float, dimensions: [Some(2), None, Some(12), None, Some(64)] }
    // Output { name: "present_8", output_type: Float, dimensions: [Some(2), None, Some(12), None, Some(64)] }
    // Output { name: "present_9", output_type: Float, dimensions: [Some(2), None, Some(12), None, Some(64)] }
    // Output { name: "present_10", output_type: Float, dimensions: [Some(2), None, Some(12), None, Some(64)] }
    // Output { name: "present_11", output_type: Float, dimensions: [Some(2), None, Some(12), None, Some(64)] }] }
    println!("{:#?}", session.inputs);
    info!("{:#?}", session.outputs);
    get_example_input(vec![
        "best hotel in bay area".to_string(),
        "here is an example of gpt2 model".to_string(),
    ]);
}

fn get_tokenizer() -> Result<tokenizers::Tokenizer> {
    Tokenizer::from_pretrained("gpt2", None)
}

fn get_example_input(
    texts: Vec<String>,
) -> (
    Vec<Array1<i64>>,
    Vec<Array1<f32>>,
    Vec<Array1<i64>>,
    Vec<ArrayD<f32>>,
) {
    let tokenizer = get_tokenizer().unwrap();
    let encodings_dict = tokenizer.encode_batch(texts, false).unwrap();

    // declare tensor
    let mut input_ids = vec![];
    let mut attention_mask = vec![];
    let mut max_length = 0;
    for encoding in &encodings_dict {
        let ids = encoding.get_ids().to_vec();
        max_length = max(max_length, ids.len());
        input_ids.push(ids);
        attention_mask.push(encoding.get_attention_mask().to_vec());
    }

    // padding
    let padded_input_ids = input_ids
        .into_iter()
        .map(|mut input_id| {
            let difference = max_length - input_id.len();
            for _i in 0..difference {
                // 50256: see get_example_input() in
                // https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/tools/transformers/notebooks/Inference_GPT2_with_OnnxRuntime_on_CPU.ipynb
                input_id.insert(0, 50256);
            }
            Array1::from_vec(input_id.into_iter().map(|v| v as i64).collect::<Vec<i64>>())
        })
        .collect::<Vec<Array1<i64>>>();

    let padded_attention_mask = attention_mask
        .into_iter()
        .map(|mut mask| {
            let difference = max_length - mask.len();
            for _i in 0..difference {
                // 0: see get_example_input() in
                // https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/tools/transformers/notebooks/Inference_GPT2_with_OnnxRuntime_on_CPU.ipynb
                mask.insert(0, 0);
            }
            Array1::from_vec(mask.into_iter().map(|v| v as f32).collect::<Vec<f32>>())
        })
        .collect::<Vec<Array1<f32>>>();

    let mut sum: i64 = 0;
    // padding and max(cumsum - 1, 0)
    let padded_position_ids = padded_attention_mask
        .clone()
        .into_iter()
        .map(|position_id| {
            position_id
                .into_iter()
                .map(|v| v as i64)
                .collect::<Vec<i64>>()
        })
        .map(|mut position_id| {
            for item in &mut position_id {
                sum += *item;
                *item = if sum > 0 { sum - 1 } else { 0 };
            }
            sum = 0;
            Array1::from_vec(position_id)
        })
        .collect::<Vec<Array1<i64>>>();

    // num_attention_heads = model.config.n_head
    // hidden_size = model.config.n_embd
    // num_layer = model.config.n_layer
    let past_shape = vec![
        2,
        1,  // batch_size,
        12, // num_attention_heads,
        0,
        768 / 12, // hidden_size / num_attention_heads,
    ];

    let mut empty_past = vec![];
    let num_layer = 12;
    for _i in 0..num_layer {
        empty_past.push(ArrayD::<f32>::from_shape_vec(past_shape.clone(), vec![]).unwrap());
    }

    println!(
        "{:?},\n {:?},\n {:?},\n {:?},",
        padded_input_ids, padded_attention_mask, padded_position_ids, empty_past
    );
    (
        padded_input_ids,
        padded_attention_mask,
        padded_position_ids,
        empty_past,
    )
}
