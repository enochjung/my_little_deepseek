use super::{Format, parse_hex_u32, parse_u8};
use crate::storage::{Host, Storage};

pub enum UnicodeFormat<'a> {
    UnicodeCharacterDatabase { path: &'a str },
}

pub(crate) struct UCDLine {
    pub(crate) codepoint: u32,
    pub(crate) combining_class: u8,
    pub(crate) decomposition: Vec<u32>,
}

impl Format for UnicodeFormat<'_> {
    type Output<'a> = Result<UCDLine, crate::Error>;
    type Parser = for<'a> fn(&'a Host) -> Box<dyn Iterator<Item = Self::Output<'a>> + 'a>;

    fn read(&self) -> Result<(Host, Self::Parser), crate::Error> {
        match self {
            &UnicodeFormat::UnicodeCharacterDatabase { path } => {
                let file = Host::try_from(path)?;
                Ok((file, parse_ucd))
            }
        }
    }
}

fn parse_ucd(file: &Host) -> Box<dyn Iterator<Item = Result<UCDLine, crate::Error>> + '_> {
    let path = file.name();
    let lines = file.as_slice().split(|&x| x == b'\n');

    let iter = lines
        .enumerate()
        .filter(|(_, text)| !text.is_empty())
        .map(|(idx, text)| {
            let line_no = idx + 1;
            parse_ucd_line(text, path, line_no)
        });

    Box::new(iter)
}

fn parse_ucd_line(text: &[u8], path: &str, line_no: usize) -> Result<UCDLine, crate::Error> {
    let fields = text.split(|x| *x == b';').collect::<Vec<_>>();
    if fields.len() != 15 {
        return Err(crate::Error::broken_data(path, line_no));
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
