use super::{Format, parse_string_with_escape_sequence, parse_u32};
use crate::storage::{Host, Storage};

pub enum VocabFormat<'a> {
    HuggingFace { path: &'a str },
}

impl Format for VocabFormat<'_> {
    type Output<'a> = Result<(String, u32), crate::Error>;
    type Parser = for<'a> fn(&'a Host) -> Box<dyn Iterator<Item = Self::Output<'a>> + 'a>;

    fn read(&self) -> Result<(Host, Self::Parser), crate::Error> {
        match self {
            &VocabFormat::HuggingFace { path } => {
                let file = Host::try_from(path)?;
                Ok((file, parse_huggingface))
            }
        }
    }
}

fn parse_huggingface(
    file: &Host,
) -> Box<dyn Iterator<Item = Result<(String, u32), crate::Error>> + '_> {
    let path = file.name();
    let lines = file.as_slice().split(|&x| x == b'\n');

    let iter = lines
        .enumerate()
        .filter(|(_, text)| !text.is_empty() && text[0] == b' ')
        .map(|(idx, text)| {
            let line_no = idx + 1;
            let vocab_line = parse_line(text, path, line_no)?;
            Ok(vocab_line)
        });

    Box::new(iter)
}

fn parse_line(text: &[u8], path: &str, line: usize) -> Result<(String, u32), crate::Error> {
    if text.len() < 7 || text[0] != b' ' || text[1] != b' ' || text[2] != b'"' {
        return Err(crate::Error::broken_data(path, line));
    }

    let mut cqi = 3;
    while cqi < text.len() {
        if text[cqi] == b'"' {
            break;
        } else if text[cqi] == b'\\' {
            cqi += 2;
        } else {
            cqi += 1;
        }
    }

    if text.len() < cqi + 4 || text[cqi] != b'"' || text[cqi + 1] != b':' || text[cqi + 2] != b' ' {
        return Err(crate::Error::broken_data(path, line));
    }

    let id = parse_u32(&text[cqi + 3..]);
    let text = parse_string_with_escape_sequence(&text[3..cqi]);

    Ok((text, id))
}
