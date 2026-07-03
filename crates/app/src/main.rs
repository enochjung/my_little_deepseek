use backend_host::Host;
use config::{
    CompositionExclusionFormat, Configure, MergeFormat, UnicodeFormat, VocabFormat, WeightFormat,
};
use inference::Model;
use session::Session;

use std::io::Write;
use std::time::Duration;

const UNICODE_PATH: &str = "model/UnicodeData.txt";
const COMPOSITION_EXCLUSION_PATH: &str = "model/CompositionExclusions.txt";
const MERGE_PATH: &str = "model/merges.json";
const VOCAB_PATH: &str = "model/vocab.json";
const WEIGHT_PATH: &str = "model/model.safetensors";

fn main() {
    std::panic::set_hook(Box::new(|info| {
        println!("\n\n---------------------------------");
        println!("Fatal error occurred!");
        println!("{}", info);
        println!("---------------------------------");
        std::process::exit(1);
    }));

    print!("[] Initializing... ");
    std::io::stdout().flush().unwrap();

    let conf = Configure::new()
        .unicode_format(UnicodeFormat::UnicodeCharacterDatabase {
            path: UNICODE_PATH.to_string(),
        })
        .composition_exclusion_format(CompositionExclusionFormat::UnicodeCharacterDatabase {
            path: COMPOSITION_EXCLUSION_PATH.to_string(),
        })
        .merge_format(MergeFormat::HuggingFace {
            path: MERGE_PATH.to_string(),
        })
        .vocab_format(VocabFormat::HuggingFace {
            path: VOCAB_PATH.to_string(),
        })
        .weight_format(WeightFormat::Safetensor {
            path: WEIGHT_PATH.to_string(),
        });

    let model = Model::<f32, Host<f32>, Host<f32>>::new(conf).expect("Failed to initialize model");
    let mut session = Session::new(&model).expect("Failed to initialize session");

    println!("done!");
    println!("---------------------------------");

    std::thread::scope(|s| {
        loop {
            print!("[User]: ");
            std::io::stdout().flush().unwrap();

            let mut input = String::new();
            let bytes_read = std::io::stdin()
                .read_line(&mut input)
                .expect("Failed to read line from user");
            if bytes_read == 0 {
                break;
            }

            let input = input.trim_end_matches('\n');
            if input == "/exit" {
                break;
            }

            println!();

            let mut session_task = session
                .send_prompt(s, input)
                .expect("Failed to send prompt");

            println!("[Assistant]: ");

            loop {
                while let Some(output) = session_task.get_next_string() {
                    print!("{output}");
                    std::io::stdout().flush().unwrap();
                }

                if session_task.is_finished() {
                    while let Some(output) = session_task.get_next_string() {
                        print!("{output}");
                        std::io::stdout().flush().unwrap();
                    }
                    break;
                }

                std::thread::sleep(Duration::from_millis(50));
            }

            session = session_task.finish_decoding().expect("Failed to decode");

            println!();
        }
    });

    println!("---------------------------------");
    println!("Goodbye!");
}
