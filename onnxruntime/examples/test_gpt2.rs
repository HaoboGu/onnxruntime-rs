use std::{cmp::max, collections::HashMap, ffi::CString};

use ndarray::{Array, Array1, ArrayD};
use onnxruntime::{
    environment::Environment,
    error::{assert_not_null_pointer, call_ort, status_to_result},
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
    let (padded_input_ids, padded_attention_mask, padded_position_ids, empty_past) =
        get_example_input(vec![
            "best hotel in bay area".to_string(),
            "here is an example of gpt2 model".to_string(),
        ]);
    create_ort_output_buffer(&session, 2, 0, 9, 12, 768, 12, 50257);

    // where onnxruntime will write the iobind to
    let mut iobinding_ptr: *mut sys::OrtIoBinding = std::ptr::null_mut();
    let iobinding_ptr_ptr: *mut *mut sys::OrtIoBinding = &mut iobinding_ptr;

    // Create io-binding
    unsafe {
        call_ort(|ort| ort.CreateIoBinding.unwrap()(session.get_session_ptr(), iobinding_ptr_ptr))
    }
    .unwrap();

    // Bind input
    unsafe {
        // Bind input name to an ort Value ptr
        call_ort(|ort| {
            ort.BindInput.unwrap()(
                iobinding_ptr,
                CString::new("name").unwrap().into_raw() as *const i8,
                padded_input_ids,
            )
        })
    }
    .unwrap();
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
) {
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

    println!("shapes: {:?}", output_shapes);

    // Get output buffers, use ndarray as the buffer backed
    let mut output_buffer: HashMap<String, ArrayD<f32>> = HashMap::new();
    for (name, shape) in output_shapes {
        let data = ArrayD::<f32>::zeros(shape);
        output_buffer.insert(name, data);
    }
    for b in output_buffer.values() {
        println!("buffers shape: {:?}", b.shape());
    }
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
    (
        padded_input_ids,
        padded_attention_mask,
        padded_position_ids,
        empty_past,
    )
}
