use my_little_deepseek::*;

use std::io::Write;
use std::time::Duration;

const UNICODE_PATH: &'static str = "model/UnicodeData.txt";
const COMPOSITION_EXCLUSION_PATH: &'static str = "model/CompositionExclusions.txt";
const MERGE_PATH: &'static str = "model/merges.json";
const VOCAB_PATH: &'static str = "model/vocab.json";
const WEIGHT_PATH: &'static str = "model/model.safetensors";

fn main() {
    print!("[] Initializing... ");
    std::io::stdout().flush().unwrap();

    let conf = config::Configure::new()
        .unicode_format(config::UnicodeFormat::UnicodeCharacterDatabase {
            path: UNICODE_PATH.to_string(),
        })
        .composition_exclusion_format(
            config::CompositionExclusionFormat::UnicodeCharacterDatabase {
                path: COMPOSITION_EXCLUSION_PATH.to_string(),
            },
        )
        .merge_format(config::MergeFormat::HuggingFace {
            path: MERGE_PATH.to_string(),
        })
        .vocab_format(config::VocabFormat::HuggingFace {
            path: VOCAB_PATH.to_string(),
        })
        .weight_format(config::WeightFormat::Safetensor {
            path: WEIGHT_PATH.to_string(),
        });

    let model = Model::new(conf).expect("initializing model should succeed");
    let mut session = model.new_session();

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

        std::io::stdout().flush().unwrap();
        println!();

        let mut session_task = session
            .send_prompt(input)
            .expect("generating session-task should succeed");

        println!("[Assistant]: ");

        loop {
            if let Some(output) = session_task.get_next_string() {
                print!("{output}");
                std::io::stdout().flush().unwrap();
            }

            if session_task.is_finished() {
                session = session_task
                    .finish_decoding()
                    .expect("decoding should succeed");
                break;
            }

            std::thread::sleep(Duration::from_millis(100));
        }

        println!();
    }

    println!("---------------------------------");
    println!("Goodbye!");
}
