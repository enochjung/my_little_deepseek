use backend_host::Mmap;
use core::MLTError;

use super::{Format, parse_string_with_escape_sequence};

use std::fs::File;

pub enum MergeFormat {
    HuggingFace { path: String },
}

impl Format for MergeFormat {
    type Output = Result<(String, String), MLTError>;
    type Parser = fn(&Mmap<u8>) -> Box<dyn Iterator<Item = Self::Output> + '_>;

    fn read(&self) -> Result<(Mmap<u8>, Self::Parser), MLTError> {
        match &self {
            MergeFormat::HuggingFace { path } => {
                let file = File::open(path).map_err(MLTError::io)?;
                let mem = Mmap::try_from(file)?;
                Ok((mem, parse_huggingface))
            }
        }
    }
}

fn parse_huggingface(
    file: &Mmap<u8>,
) -> Box<dyn Iterator<Item = Result<(String, String), MLTError>> + '_> {
    let lines = file.as_slice().split(|&x| x == b'\n');

    let iter = lines
        .enumerate()
        .filter(|(_, text)| !text.is_empty() && text[0] == b' ')
        .map(|(idx, text)| {
            let line_no = idx + 1;
            let merge_line = parse_line(text, line_no)?;
            Ok(merge_line)
        });

    Box::new(iter)
}

fn parse_line(text: &[u8], line: usize) -> Result<(String, String), MLTError> {
    if text.len() < 7 || text[0] != b' ' || text[1] != b' ' || text[2] != b'"' {
        return Err(MLTError::broken_data(line));
    }

    let mut cqi = 3;
    let mut si = 0;
    while cqi < text.len() {
        if text[cqi] == b'"' {
            break;
        } else if text[cqi] == b'\\' {
            cqi += 1;
        } else if text[cqi] == b' ' {
            if si == 0 {
                si = cqi;
            } else {
                return Err(MLTError::broken_data(line));
            }
        }
        cqi += 1;
    }

    if text.len() < cqi + 1 || text[cqi] != b'"' || si == 0 {
        return Err(MLTError::broken_data(line));
    }

    let left = parse_string_with_escape_sequence(&text[3..si]);
    let right = parse_string_with_escape_sequence(&text[si + 1..cqi]);

    Ok((left, right))
}
