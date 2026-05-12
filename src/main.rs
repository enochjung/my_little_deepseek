use my_little_deepseek::*;

use std::io::Write;

const UNICODE_PATH: &'static str = "model/UnicodeData.txt";
const COMPOSITION_EXCLUSION_PATH: &'static str = "model/CompositionExclusions.txt";
const MERGE_PATH: &'static str = "model/merges.json";
const VOCAB_PATH: &'static str = "model/vocab.json";
const WEIGHT_PATH: &'static str = "model/model.safetensors";

fn main() {
    print!("[] Initializing... ");
    std::io::stdout().flush().unwrap();

    let conf = config::Configure::new()
        .unicode_format(config::UnicodeFormat::UnicodeCharacterDatabase { path: UNICODE_PATH })
        .composition_exclusion_format(
            config::CompositionExclusionFormat::UnicodeCharacterDatabase {
                path: COMPOSITION_EXCLUSION_PATH,
            },
        )
        .merge_format(config::MergeFormat::HuggingFace { path: MERGE_PATH })
        .vocab_format(config::VocabFormat::HuggingFace { path: VOCAB_PATH })
        .weight_format(config::WeightFormat::Safetensor { path: WEIGHT_PATH });

    let model = Model::new(conf).expect("initializing model should succeed");
    let mut session = Session::new(&model).expect("generating new session should succeed");

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

        let output = session
            .send_prompt(input)
            .expect("inferencing should succeed");

        println!("done!");

        println!("[Assistant]: {output}");
    }

    println!("---------------------------------");
    println!("Goodbye!");
}
