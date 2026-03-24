use super::{Binary, Error, Text, parse_string_with_escape_sequence};
use crate::inference::utils;
use std::fs::File;

pub struct MergeText {
    path: String,
    mmap: utils::Mmap,
}

impl MergeText {
    pub fn new(path: &str) -> Result<Self, Error> {
        let file = File::open(path).map_err(|err| Error::io(path, err))?;
        let mmap = utils::Mmap::new(&file).map_err(|err| Error::io(path, err))?;

        Ok(Self {
            path: path.to_string(),
            mmap,
        })
    }
}

impl Text for MergeText {
    type Output<'a> = Box<dyn Iterator<Item = Result<(String, String), Error>> + 'a>;

    fn parse(&self) -> Result<Self::Output<'_>, Error> {
        let lines = self.mmap.as_slice().split(|&x| x == b'\n');

        let iter = lines
            .enumerate()
            .filter(|(_, text)| !text.is_empty() && text[0] == b' ')
            .map(|(idx, text)| {
                let line_no = idx + 1;
                let merge_line = parse_line(text, &self.path, line_no)?;
                Ok(merge_line)
            });

        Ok(Box::new(iter))
    }
}

fn parse_line(text: &[u8], path: &str, line: usize) -> Result<(String, String), Error> {
    if text.len() < 7 || text[0] != b' ' || text[1] != b' ' || text[2] != b'"' {
        return Err(Error::broken_data(path, line));
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
                return Err(Error::broken_data(path, line));
            }
        }
        cqi += 1;
    }

    if text.len() < cqi + 1 || text[cqi] != b'"' || si == 0 {
        return Err(Error::broken_data(path, line));
    }

    let left = parse_string_with_escape_sequence(&text[3..si]);
    let right = parse_string_with_escape_sequence(&text[si + 1..cqi]);

    Ok((left, right))
}

#[allow(unused)]
pub struct MergeBinary {
    path: String,
    mmap: utils::Mmap,
}

impl MergeBinary {
    pub fn new(path: &str) -> Result<Self, Error> {
        let file = File::open(path).map_err(|err| Error::io(path, err))?;
        let mmap = utils::Mmap::new(&file).map_err(|err| Error::io(path, err))?;

        Ok(Self {
            path: path.to_string(),
            mmap,
        })
    }
}

impl Binary for MergeBinary {
    fn raw(&self) -> Result<&[u8], Error> {
        Ok(self.mmap.as_slice())
    }
}
