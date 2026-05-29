use super::{Format, parse_hex_u32};
use crate::storage::Mmap;

pub enum CompositionExclusionFormat<'a> {
    UnicodeCharacterDatabase { path: &'a str },
}

impl Format for CompositionExclusionFormat<'_> {
    type Output<'a> = u32;
    type Parser = for<'a> fn(&'a Mmap) -> Box<dyn Iterator<Item = Self::Output<'a>> + 'a>;

    fn read(&self) -> Result<(Mmap, Self::Parser), crate::Error> {
        match self {
            &CompositionExclusionFormat::UnicodeCharacterDatabase { path } => {
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
