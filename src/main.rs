use std::fs::File;

mod error;
mod inference_app;
mod tokenizer;

use inference_app::InferenceApp;

const UNICODE_DATA_PATH: &'static str = "model/unicode_data.txt";
const COMPOSITION_EXCLUSIONS_PATH: &'static str = "model/composition_exclusions.txt";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let unicode_data_file = File::open(UNICODE_DATA_PATH)?;
    let composition_exclusions_file = File::open(COMPOSITION_EXCLUSIONS_PATH)?;

    #[allow(unused)]
    let app = InferenceApp::new(unicode_data_file, composition_exclusions_file)?;

    Ok(())
}
