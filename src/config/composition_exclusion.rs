use super::{Format, parse_hex_u32};
use crate::storage::Mmap;

pub enum CompositionExclusionFormat {
    UnicodeCharacterDatabase { path: String },
}

impl Format for CompositionExclusionFormat {
    type Output = u32;
    type Parser = fn(&Mmap) -> Box<dyn Iterator<Item = Self::Output> + '_>;

    fn read(&self) -> Result<(Mmap, Self::Parser), crate::Error> {
        match &self {
            CompositionExclusionFormat::UnicodeCharacterDatabase { path } => {
                let file = Mmap::new(path)?;
                Ok((file, parse_ucd))
            }
        }
    }
}

fn parse_ucd(file: &Mmap) -> Box<dyn Iterator<Item = u32> + '_> {
    let lines = file.as_slice().split(|&x| x == b'\n');

    let iter = lines
        .filter(|text| !text.is_empty() && text[0] != b'#')
        .map(|text| parse_hex_u32(text));

    Box::new(iter)
}
