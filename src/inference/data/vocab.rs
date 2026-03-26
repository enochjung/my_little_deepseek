use super::{Binary, Error, Text, parse_string_with_escape_sequence, parse_u32};
use crate::inference::utils;
use std::fs::File;

pub struct VocabText {
    path: String,
    mmap: utils::Mmap,
}

impl VocabText {
    pub fn new(path: &str) -> Result<Self, Error> {
        let file = File::open(path).map_err(|err| Error::io(path, err))?;
        let mmap = utils::Mmap::new(&file).map_err(|err| Error::io(path, err))?;

        Ok(Self {
            path: path.to_string(),
            mmap,
        })
    }
}

impl Text for VocabText {
    type Output<'a> = Box<dyn Iterator<Item = Result<(String, u32), Error>> + 'a>;

    fn parse(&self) -> Result<Self::Output<'_>, Error> {
        let lines = self.mmap.as_slice().split(|&x| x == b'\n');

        let iter = lines
            .enumerate()
            .filter(|(_, text)| !text.is_empty() && text[0] == b' ')
            .map(|(idx, text)| {
                let line_no = idx + 1;
                let vocab_line = parse_line(text, &self.path, line_no)?;
                Ok(vocab_line)
            });

        Ok(Box::new(iter))
    }
}

fn parse_line(text: &[u8], path: &str, line: usize) -> Result<(String, u32), Error> {
    if text.len() < 7 || text[0] != b' ' || text[1] != b' ' || text[2] != b'"' {
        return Err(Error::broken_data(path, line));
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
        return Err(Error::broken_data(path, line));
    }

    let id = parse_u32(&text[cqi + 3..]);
    let text = parse_string_with_escape_sequence(&text[3..cqi]);

    Ok((text, id))
}

pub struct VocabBinary {
    #[allow(unused)]
    path: String,
    mmap: utils::Mmap,
}

impl VocabBinary {
    pub fn new(path: &str) -> Result<Self, Error> {
        let file = File::open(path).map_err(|err| Error::io(path, err))?;
        let mmap = utils::Mmap::new(&file).map_err(|err| Error::io(path, err))?;

        Ok(Self {
            path: path.to_string(),
            mmap,
        })
    }
}

#[allow(unused)]
impl Binary for VocabBinary {
    fn raw(&self) -> Result<&[u8], Error> {
        Ok(self.mmap.as_slice())
    }
}
