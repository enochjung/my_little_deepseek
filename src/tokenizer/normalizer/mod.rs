use std::fs::File;

use crate::error::AppError;

mod nfc;
mod nfd;
mod table;

use table::{CombiningClassMap, CompositionExclusions, CompositionMap, DecompositionMap};

// Hangul-specific algorithm is not implemented.
pub struct NormalizerEngine {
    decomposition_map: DecompositionMap,
    combining_class_map: CombiningClassMap,
    composition_map: CompositionMap,
    composition_exclusions: CompositionExclusions,
}

impl NormalizerEngine {
    pub fn new(
        unicode_data_file: File,
        composition_exclusions_file: File,
    ) -> Result<Self, AppError> {
        let (decomposition_map, combining_class_map, composition_map, composition_exclusions) =
            table::new(unicode_data_file, composition_exclusions_file)?;

        Ok(Self {
            decomposition_map,
            combining_class_map,
            composition_map,
            composition_exclusions,
        })
    }

    #[allow(unused)]
    pub fn normalize(&self, context: &str) -> Result<String, AppError> {
        if context.is_ascii() {
            return Ok(context.to_owned());
        }

        let codepoints: Vec<u32> = context.chars().map(u32::from).collect();
        let decomposed = nfd::canonical_decompose(&codepoints, &self.decomposition_map);
        let ordered = nfd::reorder_by_canonical_class(&decomposed, &self.combining_class_map);
        let recomposed = nfc::recompose(
            &ordered,
            &self.combining_class_map,
            &self.composition_map,
            &self.composition_exclusions,
        );

        let mut out = String::with_capacity(context.len());
        for codepoint in recomposed {
            let Some(ch) = char::from_u32(codepoint) else {
                return Err(AppError::InvalidState(
                    "normalization produced invalid Unicode scalar value",
                ));
            };
            out.push(ch);
        }

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_engine() -> NormalizerEngine {
        let unicode_data_path = format!(
            "{}/src/tokenizer/normalizer/NFC/UnicodeData.txt",
            env!("CARGO_MANIFEST_DIR")
        );
        let composition_exclusions_path = format!(
            "{}/src/tokenizer/normalizer/NFC/CompositionExclusions.txt",
            env!("CARGO_MANIFEST_DIR")
        );

        let unicode_data_file = File::open(unicode_data_path).expect("open UnicodeData.txt");
        let composition_exclusions_file =
            File::open(composition_exclusions_path).expect("open CompositionExclusions.txt");

        NormalizerEngine::new(unicode_data_file, composition_exclusions_file)
            .expect("build normalizer engine")
    }

    #[test]
    fn normalize_embedded_vectors() {
        let engine = build_engine();
        let cases = [
            ("Hello!", "Hello!"),
            (
                "Summarize: Rust ownership prevents data races.",
                "Summarize: Rust ownership prevents data races.",
            ),
            ("What is 2 + 2?", "What is 2 + 2?"),
            (
                "Whitespace test:  keep   multiple spaces, tabs\t, and blank lines\n\nend.",
                "Whitespace test:  keep   multiple spaces, tabs\t, and blank lines\n\nend.",
            ),
            (
                "Emoji test: cats 😺 rockets 🚀 and sparkles ✨.",
                "Emoji test: cats 😺 rockets 🚀 and sparkles ✨.",
            ),
            (
                "Answer with JSON: { \"name\": \"Alice\", \"age\": 27 }",
                "Answer with JSON: { \"name\": \"Alice\", \"age\": 27 }",
            ),
            (
                "List these: 2, 3, 5, 7, 11, 13, 17, 19",
                "List these: 2, 3, 5, 7, 11, 13, 17, 19",
            ),
            (
                "Code tip: avoid unwrap() in production Rust.",
                "Code tip: avoid unwrap() in production Rust.",
            ),
            ("Compute: 127 * 43 = 5461", "Compute: 127 * 43 = 5461"),
            (
                "Code:\n```python\nfor i in range(3):\n    print(i)\n```",
                "Code:\n```python\nfor i in range(3):\n    print(i)\n```",
            ),
        ];

        for (input, expected) in cases {
            let actual = engine.normalize(input).expect("normalize should succeed");
            assert_eq!(actual, expected, "normalize mismatch for input: {input}");
        }
    }
}
