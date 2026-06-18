use backend_host::Mmap;
use core::MLTError;

use super::{Format, parse_hex_u32, parse_u8};

use std::fs::File;

pub enum UnicodeFormat {
    UnicodeCharacterDatabase { path: String },
}

pub struct UCDLine {
    pub codepoint: u32,
    pub combining_class: u8,
    pub decomposition: Vec<u32>,
}

impl Format for UnicodeFormat {
    type Output = Result<UCDLine, MLTError>;
    type Parser = fn(&Mmap<u8>) -> Box<dyn Iterator<Item = Self::Output> + '_>;

    fn read(&self) -> Result<(Mmap<u8>, Self::Parser), MLTError> {
        match &self {
            UnicodeFormat::UnicodeCharacterDatabase { path } => {
                let file = File::open(path).map_err(MLTError::io)?;
                let mem = Mmap::try_from(file)?;
                Ok((mem, parse_ucd))
            }
        }
    }
}

fn parse_ucd(file: &Mmap<u8>) -> Box<dyn Iterator<Item = Result<UCDLine, MLTError>> + '_> {
    let lines = file.as_slice().split(|&x| x == b'\n');

    let iter = lines
        .enumerate()
        .filter(|(_, text)| !text.is_empty())
        .map(|(idx, text)| {
            let line_no = idx + 1;
            parse_ucd_line(text, line_no)
        });

    Box::new(iter)
}

fn parse_ucd_line(text: &[u8], line_no: usize) -> Result<UCDLine, MLTError> {
    let fields = text.split(|x| *x == b';').collect::<Vec<_>>();
    if fields.len() != 15 {
        return Err(MLTError::broken_data(line_no));
    }

    let codepoint = parse_hex_u32(fields[0]);
    let combining_class = parse_u8(fields[3]);
    let decomposition = if fields[5].is_empty() || fields[5][0] == b'<' {
        Vec::new()
    } else {
        fields[5].split(|x| *x == b' ').map(parse_hex_u32).collect()
    };

    Ok(UCDLine {
        codepoint,
        combining_class,
        decomposition,
    })
}
