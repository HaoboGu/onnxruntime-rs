use std::{borrow::BorrowMut, cmp::max, collections::HashMap, ffi::CString};

use ndarray::{Array, Array1, Array2, ArrayD, Ix, IxDyn};
use onnxruntime::{
    environment::Environment,
    error::{assert_not_null_pointer, call_ort, status_to_result},
    iobinding,
    session::Session,
    tensor::{self, OrtOwnedTensor, OrtTensor},
    GraphOptimizationLevel,
};
use onnxruntime_sys as sys;
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
    let env = Environment::builder()
        .with_name("env")
        .with_log_level(onnxruntime::LoggingLevel::Info)
        .build()
        .unwrap();
    let mut session = env
        .new_session_builder()
        .unwrap()
        .with_optimization_level(GraphOptimizationLevel::All)
        .unwrap()
        .with_model_from_file(
            // "/Users/haobogu/Projects/rust/onnxruntime-rs/onnxruntime/examples/gpt2.onnx",
            "D:\\Projects\\onnxruntime-rs\\onnxruntime\\examples\\gpt2.onnx",
        )
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
    // println!("{:#?}", session.inputs);
    // info!("{:#?}", session.outputs);
    let (mut padded_input_ids, mut padded_attention_mask, mut padded_position_ids, mut empty_past) =
        get_example_input(vec![
            "best hotel in bay area".to_string(),
            "here is an example of gpt2 model".to_string(),
        ]);
    let mut output_buffer = create_ort_output_buffer(&session, 2, 0, 9, 12, 768, 12, 50257);

    // Create io bindings
    let io_binding = iobinding::IoBinding::new(session.session_ptr).unwrap();

    // Bind values

    println!("input ids shape: {:?}", padded_input_ids.shape());
    println!("attention mask shape: {:?}", padded_attention_mask.shape());
    println!("position ids shape: {:?}", padded_position_ids.shape());
    io_binding.bind_input(&session, "input_ids", &mut padded_input_ids);
    io_binding.bind_input(&session, "attention_mask", &mut padded_attention_mask);
    io_binding.bind_input(&session, "position_ids", &mut padded_position_ids);
    for (j, past) in empty_past.iter_mut().enumerate() {
        println!("past {} shape: {:?}", j, past.shape());

        // Here we use the pointer of padded_input_ids
        // See: https://github.com/microsoft/onnxruntime/blob/3ec3e9f70534ca13823add8bfc75722f07d67253/onnxruntime/python/tools/transformers/gpt2_helper.py#L508
        io_binding.bind_input(
            &session,
            &("past_".to_string() + &j.to_string()),
            &mut padded_input_ids,
        );
    }
    for (name, buf) in &mut output_buffer {
        println!("output {}'s shape: {:?}", name, buf.shape());
        io_binding.bind_output(&session, &name, buf);
    }

    println!("complete binding");
    session.run_with_iobinding(io_binding);
    // TODO: get output from iobinding buffer
    // get_outputs_from_io_binding_buffer();
    println!("present0:\n{:?}", output_buffer["present_0"]);
    println!("logits:\n{:?}", output_buffer["present_0"]);
    // let result = get_outputs_from_io_binding_buffer(&session, output_buffer, vec![]);
}

fn get_outputs_from_io_binding_buffer(
    session: &Session,
    output_buffer: HashMap<String, ArrayD<f32>>,
    output_shapes: Vec<usize>,
) -> HashMap<String, ArrayD<f32>> {
    return output_buffer.clone();
}

fn create_ort_output_buffer(
    session: &Session,
    batch_size: usize,
    past_sequence_size: usize,
    sequence_size: usize,
    num_attention_heads: usize,
    hidden_layer_size: usize,
    num_hidden_layer: usize,
    vocab_size: usize,
) -> HashMap<String, ArrayD<f32>> {
    // Get output_shapes first
    let mut output_shapes: HashMap<String, Vec<usize>> = HashMap::new();
    for output in &session.outputs {
        let output_size = if output.name == "logits" {
            vocab_size
        } else {
            hidden_layer_size
        };
        let last_state_shape = [batch_size, sequence_size, output_size];
        let present_state_shape = [
            2,
            batch_size,
            num_attention_heads,
            past_sequence_size + sequence_size,
            hidden_layer_size / num_attention_heads,
        ];
        output_shapes.insert(output.name.clone(), last_state_shape.to_vec());
        for i in 0..num_hidden_layer {
            let mut k = String::from("present_");
            k.push_str(&i.to_string());
            output_shapes.insert(k, present_state_shape.to_vec());
        }
    }

    // Get output buffers, use ndarray as the buffer backed
    let mut output_buffer: HashMap<String, ArrayD<f32>> = HashMap::new();
    for (name, shape) in output_shapes {
        let data = ArrayD::<f32>::zeros(shape);
        output_buffer.insert(name, data);
    }
    output_buffer
}

fn get_tokenizer() -> Result<tokenizers::Tokenizer> {
    Tokenizer::from_pretrained("gpt2", None)
}

fn get_example_input(
    texts: Vec<String>,
) -> (ArrayD<i64>, ArrayD<f32>, ArrayD<i64>, Vec<ArrayD<f32>>) {
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
            input_id.into_iter().map(|v| v as i64).collect::<Vec<i64>>()
        })
        .collect::<Vec<Vec<i64>>>();

    let padded_attention_mask = attention_mask
        .into_iter()
        .map(|mut mask| {
            let difference = max_length - mask.len();
            for _i in 0..difference {
                // 0: see get_example_input() in
                // https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/tools/transformers/notebooks/Inference_GPT2_with_OnnxRuntime_on_CPU.ipynb
                mask.insert(0, 0);
            }
            mask.into_iter().map(|v| v as f32).collect::<Vec<f32>>()
        })
        .collect::<Vec<Vec<f32>>>();

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
            position_id
        })
        .collect::<Vec<Vec<i64>>>();

    // num_attention_heads = model.config.n_head
    // hidden_size = model.config.n_embd
    // num_layer = model.config.n_layer
    let past_shape = vec![
        2,
        2,  // batch_size,
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

    // Flat all
    let padded_input_ids_array = ArrayD::<i64>::from_shape_vec(
        IxDyn(&[2, max_length]),
        padded_input_ids
            .into_iter()
            .flatten()
            .into_iter()
            .collect::<Vec<i64>>(),
    )
    .unwrap();

    let padded_attention_mask_array = ArrayD::<f32>::from_shape_vec(
        IxDyn(&[2, max_length]),
        padded_attention_mask
            .into_iter()
            .flatten()
            .into_iter()
            .collect::<Vec<f32>>(),
    )
    .unwrap();

    let padded_position_ids_array = ArrayD::<i64>::from_shape_vec(
        IxDyn(&[2, max_length]),
        padded_position_ids
            .into_iter()
            .flatten()
            .into_iter()
            .collect::<Vec<i64>>(),
    )
    .unwrap();
    (
        padded_input_ids_array,
        padded_attention_mask_array,
        padded_position_ids_array,
        empty_past,
    )
}
