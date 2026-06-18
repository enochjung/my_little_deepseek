use backend_host::Mmap;
use core::MLTError;

use super::{Format, parse_string_with_escape_sequence, parse_u32};

use std::fs::File;

pub enum VocabFormat {
    HuggingFace { path: String },
}

impl Format for VocabFormat {
    type Output = Result<(String, u32), MLTError>;
    type Parser = fn(&Mmap<u8>) -> Box<dyn Iterator<Item = Self::Output> + '_>;

    fn read(&self) -> Result<(Mmap<u8>, Self::Parser), MLTError> {
        match &self {
            VocabFormat::HuggingFace { path } => {
                let file = File::open(path).map_err(MLTError::io)?;
                let mem = Mmap::try_from(file)?;
                Ok((mem, parse_huggingface))
            }
        }
    }
}

fn parse_huggingface(
    file: &Mmap<u8>,
) -> Box<dyn Iterator<Item = Result<(String, u32), MLTError>> + '_> {
    let lines = file.as_slice().split(|&x| x == b'\n');

    let iter = lines
        .enumerate()
        .filter(|(_, text)| !text.is_empty() && text[0] == b' ')
        .map(|(idx, text)| {
            let line_no = idx + 1;
            let vocab_line = parse_line(text, line_no)?;
            Ok(vocab_line)
        });

    Box::new(iter)
}

fn parse_line(text: &[u8], line: usize) -> Result<(String, u32), MLTError> {
    if text.len() < 7 || text[0] != b' ' || text[1] != b' ' || text[2] != b'"' {
        return Err(MLTError::broken_data(line));
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
        return Err(MLTError::broken_data(line));
    }

    let id = parse_u32(&text[cqi + 3..]);
    let text = parse_string_with_escape_sequence(&text[3..cqi]);

    Ok((text, id))
}
