use super::{Format, parse_hex_u32, parse_u8};
use crate::device::Cpu;
use std::fs::File;

pub enum UnicodeFormat {
    UnicodeCharacterDatabase { path: String },
}

pub(crate) struct UCDLine {
    pub(crate) codepoint: u32,
    pub(crate) combining_class: u8,
    pub(crate) decomposition: Vec<u32>,
}

impl Format for UnicodeFormat {
    type Output = Result<UCDLine, crate::Error>;
    type Parser = fn(&Cpu) -> Box<dyn Iterator<Item = Self::Output> + '_>;

    fn read(&self) -> Result<(Cpu, Self::Parser), crate::Error> {
        match &self {
            UnicodeFormat::UnicodeCharacterDatabase { path } => {
                let file = File::open(path).map_err(crate::Error::io)?;
                let storage = Cpu::try_from(file)?;
                Ok((storage, parse_ucd))
            }
        }
    }
}

fn parse_ucd(file: &Cpu) -> Box<dyn Iterator<Item = Result<UCDLine, crate::Error>> + '_> {
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

fn parse_ucd_line(text: &[u8], line_no: usize) -> Result<UCDLine, crate::Error> {
    let fields = text.split(|x| *x == b';').collect::<Vec<_>>();
    if fields.len() != 15 {
        return Err(crate::Error::broken_data(line_no));
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
