use super::{Format, parse_hex_u32};
use crate::device::Cpu;
use std::fs::File;

pub enum CompositionExclusionFormat {
    UnicodeCharacterDatabase { path: String },
}

impl Format for CompositionExclusionFormat {
    type Output = u32;
    type Parser = fn(&Cpu) -> Box<dyn Iterator<Item = Self::Output> + '_>;

    fn read(&self) -> Result<(Cpu, Self::Parser), crate::Error> {
        match &self {
            CompositionExclusionFormat::UnicodeCharacterDatabase { path } => {
                let file = File::open(path).map_err(crate::Error::io)?;
                let storage = Cpu::try_from(file)?;
                Ok((storage, parse_ucd))
            }
        }
    }
}

fn parse_ucd(file: &Cpu) -> Box<dyn Iterator<Item = u32> + '_> {
    let lines = file.as_slice().split(|&x| x == b'\n');

    let iter = lines
        .filter(|text| !text.is_empty() && text[0] != b'#')
        .map(|text| parse_hex_u32(text));

    Box::new(iter)
}
