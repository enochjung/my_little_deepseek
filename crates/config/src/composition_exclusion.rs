use backend_host::Mmap;
use core::MLTError;

use super::{Format, parse_hex_u32};

use std::fs::File;

pub enum CompositionExclusionFormat {
    UnicodeCharacterDatabase { path: String },
}

impl Format for CompositionExclusionFormat {
    type Output = u32;
    type Parser = fn(&Mmap<u8>) -> Box<dyn Iterator<Item = Self::Output> + '_>;

    fn read(&self) -> Result<(Mmap<u8>, Self::Parser), MLTError> {
        match &self {
            CompositionExclusionFormat::UnicodeCharacterDatabase { path } => {
                let file = File::open(path).map_err(MLTError::io)?;
                let mem = Mmap::try_from(file)?;
                Ok((mem, parse_ucd))
            }
        }
    }
}

fn parse_ucd(file: &Mmap<u8>) -> Box<dyn Iterator<Item = u32> + '_> {
    let lines = file.as_slice().split(|&x| x == b'\n');

    let iter = lines
        .filter(|text| !text.is_empty() && text[0] != b'#')
        .map(parse_hex_u32);

    Box::new(iter)
}
