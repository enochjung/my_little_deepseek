use super::{Format, parse_string_with_escape_sequence};
use crate::storage::Host;

pub enum MergeFormat<'a> {
    HuggingFace { path: &'a str },
}

impl Format for MergeFormat<'_> {
    type Output<'a> = Result<(String, String), crate::Error>;
    type Parser = for<'a> fn(&'a Host) -> Box<dyn Iterator<Item = Self::Output<'a>> + 'a>;

    fn read(&self) -> Result<(Host, Self::Parser), crate::Error> {
        match self {
            &MergeFormat::HuggingFace { path } => {
                let file = Host::try_from(path)?;
                Ok((file, parse_huggingface))
            }
        }
    }
}

fn parse_huggingface(
    file: &Host,
) -> Box<dyn Iterator<Item = Result<(String, String), crate::Error>> + '_> {
    let path = file.name();
    let lines = file.as_slice().split(|&x| x == b'\n');

    let iter = lines
        .enumerate()
        .filter(|(_, text)| !text.is_empty() && text[0] == b' ')
        .map(|(idx, text)| {
            let line_no = idx + 1;
            let merge_line = parse_line(text, path, line_no)?;
            Ok(merge_line)
        });

    Box::new(iter)
}

fn parse_line(text: &[u8], path: &str, line: usize) -> Result<(String, String), crate::Error> {
    if text.len() < 7 || text[0] != b' ' || text[1] != b' ' || text[2] != b'"' {
        return Err(crate::Error::broken_data(path, line));
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
                return Err(crate::Error::broken_data(path, line));
            }
        }
        cqi += 1;
    }

    if text.len() < cqi + 1 || text[cqi] != b'"' || si == 0 {
        return Err(crate::Error::broken_data(path, line));
    }

    let left = parse_string_with_escape_sequence(&text[3..si]);
    let right = parse_string_with_escape_sequence(&text[si + 1..cqi]);

    Ok((left, right))
}
