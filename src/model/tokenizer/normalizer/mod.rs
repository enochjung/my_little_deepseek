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

    pub fn normalize(&self, input: &str) -> Result<String, AppError> {
        if input.is_ascii() {
            return Ok(input.to_owned());
        }

        let codepoints: Vec<u32> = input.chars().map(u32::from).collect();
        let decomposed = nfd::canonical_decompose(&codepoints, &self.decomposition_map);
        let ordered = nfd::reorder_by_canonical_class(&decomposed, &self.combining_class_map);
        let recomposed = nfc::recompose(
            &ordered,
            &self.combining_class_map,
            &self.composition_map,
            &self.composition_exclusions,
        );

        let mut out = String::with_capacity(input.len());
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

    const UNICODE_DATA_PATH: &'static str = "model/UnicodeData.txt";
    const COMPOSITION_EXCLUSIONS_PATH: &'static str = "model/CompositionExclusions.txt";

    fn build_engine() -> NormalizerEngine {
        let unicode_data_file = File::open(UNICODE_DATA_PATH).expect("open UnicodeData.txt");
        let composition_exclusions_file =
            File::open(COMPOSITION_EXCLUSIONS_PATH).expect("open CompositionExclusions.txt");

        NormalizerEngine::new(unicode_data_file, composition_exclusions_file)
            .expect("build normalizer engine")
    }

    fn assert_normalized(input: &str, expected: &str) {
        let engine = build_engine();
        let normalized = engine
            .normalize(input)
            .expect("normalization should succeed");
        assert_eq!(normalized, expected, "Normalization of {:?} failed", input);
    }

    #[test]
    fn normalize_case1_latin_acute_accent() {
        // Decomposed café with combining acute accent
        assert_normalized("café\u{0301}", "café\u{0301}");
    }

    #[test]
    fn normalize_case2_latin_combining_diacritics() {
        // French école with combining diacritics
        assert_normalized("école", "école");
    }

    #[test]
    fn normalize_case3_greek_tonos() {
        // Greek Ελληνικά with tonos (stress mark)
        assert_normalized("Ελληνικά", "Ελληνικά");
    }

    #[test]
    fn normalize_case4_german_umlauts() {
        // German München with umlaut
        assert_normalized("München", "München");
    }

    #[test]
    fn normalize_case5_vietnamese_tone_marks() {
        // Vietnamese "Tiếng Việt" with tone marks
        assert_normalized("Tiếng Việt", "Tiếng Việt");
    }

    #[test]
    fn normalize_case6_devanagari_combining() {
        // Devanagari नमस्तेि with combining marks
        assert_normalized("नमस्तेि", "नमस्तेि");
    }

    #[test]
    fn normalize_case7_arabic_diacritics() {
        // Arabic السَّلَام with fathah marks
        assert_normalized("السَّلَام", "السَّلَام");
    }

    #[test]
    fn normalize_case8_hebrew_combining_marks() {
        // Hebrew שָׁלוֹם with various combining marks
        assert_normalized("שָׁלוֹם", "שָׁלוֹם");
    }

    #[test]
    fn normalize_case9_latin_ligatures() {
        // Latin ligature ﬁnance (fi ligature)
        assert_normalized("ﬁnance", "ﬁnance");
    }

    #[test]
    fn normalize_case10_mixed_composed_characters() {
        // Mixed scripts: "Café: ñoño" with multiple composed characters (é, ñ)
        assert_normalized("Café: ñoño", "Café: ñoño");
    }
}
