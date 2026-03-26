mod inference;

use std::io::Write;

const UNICODE_PATH: &'static str = "model/UnicodeData.txt";
const EXCLUSION_PATH: &'static str = "model/CompositionExclusions.txt";
const MERGE_PATH: &'static str = "model/merges.json";
const VOCAB_PATH: &'static str = "model/vocab.json";
const WEIGHT_PATH: &'static str = "model/model.safetensors";

fn main() {
    print!("[] Initializing... ");
    std::io::stdout().flush().unwrap();

    let model_data = inference::ModelData::new(
        UNICODE_PATH,
        EXCLUSION_PATH,
        MERGE_PATH,
        VOCAB_PATH,
        WEIGHT_PATH,
    )
    .expect("initializing model data should succeed");
    let mut inference_engine = inference::InferenceEngine::new(&model_data)
        .expect("initializing inference engine should succeed");

    println!("done!");
    println!("---------------------------------");

    loop {
        print!("[User]: ");
        let mut input = String::new();
        let bytes_read = std::io::stdin()
            .read_line(&mut input)
            .expect("reading line from user failed");
        if bytes_read == 0 {
            break;
        }

        let input = input.trim_end_matches('\n');
        if input == "/exit" {
            break;
        }

        print!("[] Inferencing... ");
        std::io::stdout().flush().unwrap();

        let output = inference_engine
            .run_prompt(input)
            .expect("inferencing should succeed");

        println!("done!");

        println!("[Assistant]: {output}");
    }

    println!("---------------------------------");
    println!("Goodbye!");
}
