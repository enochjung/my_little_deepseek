use super::{Error, ModelData};
use crate::inference::data::*;
use std::collections::{HashMap, HashSet};

// Hangul-specific algorithm is not implemented.
pub struct NormalizerEngine<'a> {
    #[allow(unused)]
    model_data: &'a ModelData,
    decomposition_map: DecompositionMap,
    combining_class_map: CombiningClassMap,
    composition_map: CompositionMap,
    exclusion_map: ExclusionMap,
}

impl<'a> NormalizerEngine<'a> {
    pub fn new(model_data: &'a ModelData) -> Result<Self, Error> {
        let mut decomposition_map = DecompositionMap::new();
        let mut combining_class_map = CombiningClassMap::new();
        let mut composition_map = CompositionMap::new();
        let mut exclusion_map = ExclusionMap::new();

        match (&model_data.unicode_binary, &model_data.unicode_text) {
            (Some(unicode_binary), _) => parse_unicode_binary(
                unicode_binary,
                &mut decomposition_map,
                &mut combining_class_map,
                &mut composition_map,
            ),
            (None, Some(unicode_text)) => parse_unicode_text(
                unicode_text,
                &mut decomposition_map,
                &mut combining_class_map,
                &mut composition_map,
            ),
            (None, None) => return Err(Error::data_not_provided("unicode")),
        }?;

        match (&model_data.exclusion_binary, &model_data.exclusion_text) {
            (Some(exclusion_binary), _) => {
                parse_exclusion_binary(exclusion_binary, &mut exclusion_map)
            }
            (None, Some(exclusion_text)) => {
                parse_exclusion_text(exclusion_text, &mut exclusion_map)
            }
            (None, None) => return Err(Error::data_not_provided("exclusion")),
        }?;

        Ok(Self {
            model_data,
            decomposition_map,
            combining_class_map,
            composition_map,
            exclusion_map,
        })
    }

    pub fn normalize(&self, input: &str) -> Result<String, Error> {
        if input.is_ascii() {
            return Ok(input.to_owned());
        }

        let codepoints: Vec<u32> = input.chars().map(u32::from).collect();
        let decomposed = self.decompose(&codepoints);
        let ordered = self.reorder(&decomposed);
        let recomposed = self.recompose(&ordered);

        recomposed
            .into_iter()
            .try_fold(String::with_capacity(input.len()), |mut out, codepoint| {
                let Some(ch) = char::from_u32(codepoint) else {
                    return Err(Error::invalid_char(codepoint));
                };
                out.push(ch);
                Ok(out)
            })
    }

    fn decompose(&self, input: &[u32]) -> Vec<u32> {
        let mut out = Vec::with_capacity(input.len());
        for &codepoint in input {
            self.insert_decomposed(&mut out, codepoint);
        }
        out
    }

    fn insert_decomposed(&self, out: &mut Vec<u32>, codepoint: u32) -> () {
        if let Some(decompose) = self.decomposition_map.get(codepoint) {
            for codepoint in decompose {
                self.insert_decomposed(out, *codepoint);
            }
        } else {
            out.push(codepoint);
        }
    }

    fn reorder(&self, decomposed: &[u32]) -> Vec<u32> {
        let mut out = Vec::with_capacity(decomposed.len());
        let mut segment: Vec<u32> = Vec::new();

        for &codepoint in decomposed {
            let class_id = self.combining_class_map.get(codepoint);
            if class_id == 0 {
                self.flush_reorder_segment(&mut out, &mut segment);
            }
            segment.push(codepoint);
        }
        self.flush_reorder_segment(&mut out, &mut segment);

        out
    }

    fn flush_reorder_segment(&self, out: &mut Vec<u32>, segment: &mut Vec<u32>) -> () {
        if segment.is_empty() {
            return;
        }

        let starter = segment[0];
        out.push(starter);

        let mut marks: Vec<_> = segment
            .drain(..)
            .skip(1)
            .enumerate()
            .map(|(index, codepoint)| (self.combining_class_map.get(codepoint), index, codepoint))
            .collect();

        marks.sort_by_key(|(class_id, index, _)| (*class_id, *index));

        out.extend(marks.into_iter().map(|(_, _, codepoint)| codepoint));
    }

    fn recompose(&self, ordered: &[u32]) -> Vec<u32> {
        let mut out = Vec::with_capacity(ordered.len());
        let mut segment: Vec<u32> = Vec::new();

        for &codepoint in ordered {
            let class_id = self.combining_class_map.get(codepoint);
            if class_id == 0 {
                self.flush_recompose_segment(&mut out, &mut segment);
            }
            segment.push(codepoint);
        }
        self.flush_recompose_segment(&mut out, &mut segment);

        out
    }

    fn flush_recompose_segment(&self, out: &mut Vec<u32>, segment: &mut Vec<u32>) -> () {
        if segment.is_empty() {
            return;
        }

        if self.combining_class_map.get(segment[0]) != 0 {
            out.extend(segment.drain(..));
            return;
        }

        let mut starter = segment[0];
        let mut kept_marks: Vec<(u8, u32)> = Vec::new();

        for codepoint in segment.drain(..).skip(1) {
            let class_id = self.combining_class_map.get(codepoint);
            let blocked = kept_marks
                .last()
                .map(|(prev_id, _)| *prev_id >= class_id)
                .unwrap_or(false);

            if !blocked {
                if let Some(composed) = self.composition_map.get(starter, codepoint) {
                    if !self.exclusion_map.contains(composed) {
                        starter = composed;
                        continue;
                    }
                }
            }

            kept_marks.push((class_id, codepoint));
        }

        out.push(starter);
        out.extend(kept_marks.into_iter().map(|(_, cp)| cp));
    }
}

#[allow(unused)]
fn parse_unicode_binary(
    unicode_binary: &UnicodeBinary,
    decomposition_map: &mut DecompositionMap,
    combining_class_map: &mut CombiningClassMap,
    composition_map: &mut CompositionMap,
) -> Result<(), Error> {
    todo!("parsing unicode binary is not implemented yet");
}

fn parse_unicode_text(
    unicode_text: &UnicodeText,
    decomposition_map: &mut DecompositionMap,
    combining_class_map: &mut CombiningClassMap,
    composition_map: &mut CompositionMap,
) -> Result<(), Error> {
    let iter = unicode_text.parse()?;

    for unicode_line in iter {
        let unicode_line = unicode_line?;
        let codepoint = unicode_line.codepoint;
        let combining_class = unicode_line.combining_class;
        let decomposition = unicode_line.decomposition;

        combining_class_map.insert(codepoint, combining_class);
        if !decomposition.is_empty() {
            if decomposition.len() == 2 {
                composition_map.insert(decomposition[0], decomposition[1], codepoint);
            }
            decomposition_map.insert(codepoint, decomposition);
        }
    }

    Ok(())
}

#[allow(unused)]
fn parse_exclusion_binary(
    exclusion_binary: &ExclusionBinary,
    exclusion_map: &mut ExclusionMap,
) -> Result<(), Error> {
    todo!("parsing exclusion binary is not implemented yet");
}

fn parse_exclusion_text(
    exclusion_text: &ExclusionText,
    exclusion_map: &mut ExclusionMap,
) -> Result<(), Error> {
    let iter = exclusion_text.parse()?;

    for codepoint in iter {
        let codepoint = codepoint?;
        exclusion_map.insert(codepoint);
    }

    Ok(())
}

struct DecompositionMap {
    map: HashMap<u32, Vec<u32>>,
}

impl DecompositionMap {
    fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    fn get(&self, codepoint: u32) -> Option<&Vec<u32>> {
        self.map.get(&codepoint)
    }

    fn insert(&mut self, codepoint: u32, decomposition: Vec<u32>) -> () {
        self.map.insert(codepoint, decomposition);
    }
}

struct CombiningClassMap {
    map: HashMap<u32, u8>,
}

impl CombiningClassMap {
    fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    fn get(&self, codepoint: u32) -> u8 {
        self.map.get(&codepoint).copied().unwrap_or(0)
    }

    fn insert(&mut self, codepoint: u32, combining_class: u8) -> () {
        if codepoint != 0 {
            self.map.insert(codepoint, combining_class);
        }
    }
}

struct CompositionMap {
    map: HashMap<(u32, u32), u32>,
}

impl CompositionMap {
    fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    fn get(&self, left: u32, right: u32) -> Option<u32> {
        self.map.get(&(left, right)).copied()
    }

    fn insert(&mut self, left: u32, right: u32, composed: u32) -> () {
        self.map.insert((left, right), composed);
    }
}

struct ExclusionMap {
    map: HashSet<u32>,
}

impl ExclusionMap {
    fn new() -> Self {
        Self {
            map: HashSet::new(),
        }
    }

    fn contains(&self, codepoint: u32) -> bool {
        self.map.contains(&codepoint)
    }

    fn insert(&mut self, codepoint: u32) -> () {
        self.map.insert(codepoint);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const UNICODE_PATH: &'static str = "model/UnicodeData.txt";
    const EXCLUSION_PATH: &'static str = "model/CompositionExclusions.txt";

    fn assert_normalized(input: &str, expected: &str) {
        assert_ne!(
            input.as_bytes(),
            expected.as_bytes(),
            "Normalization of {:?} data invalid",
            input
        );
        let model_data = ModelData::new(UNICODE_PATH, EXCLUSION_PATH, "none.none", "none.none")
            .expect("Error occured at initializing data");
        let engine =
            NormalizerEngine::new(&model_data).expect("Error occured at initializing normalizer");
        let normalized = engine
            .normalize(input)
            .expect("Error occured at normalizing");
        assert_eq!(
            normalized,
            expected,
            "Normalization of {:?} failed: normalized:{:?}, expected:{:?}",
            input,
            normalized.as_bytes(),
            expected.as_bytes()
        );
    }

    #[test]
    fn normalize_case1_latin_acute_accent() {
        assert_normalized("cafe\u{0301}", "café");
    }

    #[test]
    fn normalize_case2_latin_combining_diacritics() {
        assert_normalized("e\u{0301}cole", "école");
    }

    #[test]
    fn normalize_case3_greek_tonos() {
        assert_normalized("Ελληνικα\u{0301}", "Ελληνικά");
    }

    #[test]
    fn normalize_case4_german_umlauts() {
        assert_normalized("Mu\u{0308}nchen", "München");
    }

    #[test]
    fn normalize_case5_vietnamese_tone_marks() {
        assert_normalized("Tie\u{0302}\u{0301}ng Vie\u{0323}\u{0302}t", "Tiếng Việt");
    }

    #[test]
    fn normalize_case6_devanagari_combining() {
        assert_normalized("A\u{030A}", "Å");
    }

    #[test]
    fn normalize_case7_arabic_diacritics() {
        assert_normalized("السَّلَام", "السَّلَام");
    }

    #[test]
    fn normalize_case8_hebrew_combining_marks() {
        assert_normalized("שָׁלוֹם", "שָׁלוֹם");
    }

    #[test]
    fn normalize_case9_latin_ligatures() {
        assert_normalized("\u{212B}", "Å");
    }

    #[test]
    fn normalize_case10_mixed_composed_characters() {
        assert_normalized("Cafe\u{0301}: n\u{0303}on\u{0303}o", "Café: ñoño");
    }
}
