use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};

use crate::error::AppError;

pub type Codepoint = u32;

pub struct DecompositionMap {
    inner: HashMap<Codepoint, Vec<Codepoint>>,
}

impl DecompositionMap {
    fn new() -> Self {
        Self {
            inner: HashMap::new(),
        }
    }

    pub fn get(&self, codepoint: Codepoint) -> Option<&[Codepoint]> {
        self.inner.get(&codepoint).map(Vec::as_slice)
    }

    fn insert(
        &mut self,
        codepoint: Codepoint,
        decomposition: Vec<Codepoint>,
    ) -> Option<Vec<Codepoint>> {
        self.inner.insert(codepoint, decomposition)
    }
}

pub struct CombiningClassMap {
    inner: HashMap<Codepoint, u8>,
}

impl CombiningClassMap {
    fn new() -> Self {
        Self {
            inner: HashMap::new(),
        }
    }

    pub fn get(&self, codepoint: Codepoint) -> u8 {
        self.inner.get(&codepoint).copied().unwrap_or(0)
    }

    fn insert(&mut self, codepoint: Codepoint, combining_class: u8) -> Option<u8> {
        self.inner.insert(codepoint, combining_class)
    }
}

pub struct CompositionMap {
    inner: HashMap<(Codepoint, Codepoint), Codepoint>,
}

impl CompositionMap {
    fn new() -> Self {
        Self {
            inner: HashMap::new(),
        }
    }

    pub fn get(&self, starter: Codepoint, combining_mark: Codepoint) -> Option<Codepoint> {
        self.inner.get(&(starter, combining_mark)).copied()
    }

    fn insert(
        &mut self,
        starter: Codepoint,
        combining_mark: Codepoint,
        composed: Codepoint,
    ) -> Option<Codepoint> {
        self.inner.insert((starter, combining_mark), composed)
    }
}

pub struct CompositionExclusions {
    inner: HashSet<Codepoint>,
}

impl CompositionExclusions {
    fn new() -> Self {
        Self {
            inner: HashSet::new(),
        }
    }

    pub fn contains(&self, codepoint: Codepoint) -> bool {
        self.inner.contains(&codepoint)
    }

    fn insert(&mut self, codepoint: Codepoint) -> bool {
        self.inner.insert(codepoint)
    }
}

pub fn new(
    unicode_data_file: File,
    composition_exclusions_file: File,
) -> Result<
    (
        DecompositionMap,
        CombiningClassMap,
        CompositionMap,
        CompositionExclusions,
    ),
    AppError,
> {
    let (decomposition_map, combining_class_map) = parse_unicode_data(unicode_data_file)?;
    let composition_exclusions = parse_composition_exclusions(composition_exclusions_file)?;
    let composition_map = build_composition_map(&decomposition_map, &composition_exclusions)?;

    Ok((
        decomposition_map,
        combining_class_map,
        composition_map,
        composition_exclusions,
    ))
}

fn build_composition_map(
    decomposition_map: &DecompositionMap,
    composition_exclusions: &CompositionExclusions,
) -> Result<CompositionMap, AppError> {
    let mut composition_map = CompositionMap::new();

    for (&composed, decomposition) in &decomposition_map.inner {
        if composition_exclusions.contains(composed) {
            continue;
        }

        if decomposition.len() != 2 {
            continue;
        }

        let starter = decomposition[0];
        let combining_mark = decomposition[1];
        if let Some(existing) = composition_map.insert(starter, combining_mark, composed) {
            if existing != composed {
                return Err(AppError::InvalidState(
                    "conflicting composition pair mapping in Unicode data",
                ));
            }
        }
    }

    Ok(composition_map)
}

fn parse_composition_exclusions(
    composition_exclusions_file: File,
) -> Result<CompositionExclusions, AppError> {
    let mut exclusions = CompositionExclusions::new();
    let reader = BufReader::with_capacity(64 * 1024, composition_exclusions_file);

    for (index, line_result) in reader.lines().enumerate() {
        let line_no = index + 1;
        let line = line_result?;
        let trimmed = line.trim();

        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let head = match trimmed.split_once('#') {
            Some((before, _)) => before.trim(),
            None => trimmed,
        };

        if head.is_empty() {
            continue;
        }

        let Some(codepoint_hex) = head.split_whitespace().next() else {
            return Err(AppError::InvalidCompositionExclusionLine {
                line: line_no,
                reason: "missing exclusion codepoint",
            });
        };

        let codepoint = parse_hex_codepoint(codepoint_hex, line_no)?;
        exclusions.insert(codepoint);
    }

    Ok(exclusions)
}

fn parse_unicode_data(
    unicode_data_file: File,
) -> Result<(DecompositionMap, CombiningClassMap), AppError> {
    let mut decomposition_map = DecompositionMap::new();
    let mut combining_class_map = CombiningClassMap::new();
    let reader = BufReader::with_capacity(256 * 1024, unicode_data_file);

    for (index, line_result) in reader.lines().enumerate() {
        let line_no = index + 1;
        let line = line_result?;

        if line.trim().is_empty() {
            continue;
        }

        let fields: Vec<&str> = line.split(';').collect();
        if fields.len() != 15 {
            return Err(AppError::InvalidUnicodeDataLine {
                line: line_no,
                reason: "expected exactly 15 semicolon-separated fields",
            });
        }

        let codepoint = parse_hex_codepoint(fields[0], line_no)?;
        let combining_class = parse_combining_class(fields[3], line_no)?;
        combining_class_map.insert(codepoint, combining_class);

        if let Some(canonical) = parse_canonical_decomposition(fields[5], line_no)? {
            decomposition_map.insert(codepoint, canonical);
        }
    }

    Ok((decomposition_map, combining_class_map))
}

fn parse_hex_codepoint(value: &str, line_no: usize) -> Result<Codepoint, AppError> {
    u32::from_str_radix(value, 16).map_err(|_| AppError::InvalidHexCodepoint {
        line: line_no,
        value: value.to_string(),
    })
}

fn parse_combining_class(value: &str, line_no: usize) -> Result<u8, AppError> {
    value
        .parse::<u8>()
        .map_err(|_| AppError::InvalidCombiningClass {
            line: line_no,
            value: value.to_string(),
        })
}

fn parse_canonical_decomposition(
    value: &str,
    line_no: usize,
) -> Result<Option<Vec<Codepoint>>, AppError> {
    if value.is_empty() {
        return Ok(None);
    }

    if value.starts_with('<') {
        return Ok(None);
    }

    let mut codepoints = Vec::new();
    for part in value.split_whitespace() {
        codepoints.push(parse_hex_codepoint(part, line_no)?);
    }

    if codepoints.is_empty() {
        return Err(AppError::InvalidUnicodeDataLine {
            line: line_no,
            reason: "canonical decomposition field is empty after parsing",
        });
    }

    Ok(Some(codepoints))
}
