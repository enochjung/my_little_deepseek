use std::fs::File;

mod error;
mod model;

use model::InferenceModelEngine;

const UNICODE_DATA_PATH: &'static str = "model/UnicodeData.txt";
const COMPOSITION_EXCLUSIONS_PATH: &'static str = "model/CompositionExclusions.txt";
const VOCAB_PATH: &'static str = "model/vocab.json";
const MERGES_PATH: &'static str = "model/merges.json";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let unicode_data_file = File::open(UNICODE_DATA_PATH)?;
    let composition_exclusions_file = File::open(COMPOSITION_EXCLUSIONS_PATH)?;
    let vocab_file = File::open(VOCAB_PATH)?;
    let merges_file = File::open(MERGES_PATH)?;

    let mut inference_model_engine = InferenceModelEngine::new(
        unicode_data_file,
        composition_exclusions_file,
        vocab_file,
        merges_file,
    )?;

    loop {
        let mut input = String::new();
        let bytes_read = std::io::stdin().read_line(&mut input)?;
        if bytes_read == 0 {
            break;
        }

        let input = input.trim_end_matches('\n');
        if input == "/exit" {
            break;
        }

        let output = inference_model_engine.run_prompt(input)?;
        println!("{}", output);
    }

    Ok(())
}
