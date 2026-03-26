use super::{Binary, Error, Text, parse_hex_u32, parse_u8};
use crate::inference::utils;
use std::fs::File;

pub struct UnicodeText {
    path: String,
    mmap: utils::Mmap,
}

pub struct UnicodeLine {
    pub codepoint: u32,
    pub combining_class: u8,
    pub decomposition: Vec<u32>,
}

impl UnicodeText {
    pub fn new(path: &str) -> Result<Self, Error> {
        let file = File::open(path).map_err(|err| Error::io(path, err))?;
        let mmap = utils::Mmap::new(&file).map_err(|err| Error::io(path, err))?;

        Ok(Self {
            path: path.to_string(),
            mmap,
        })
    }
}

impl Text for UnicodeText {
    type Output<'a> = Box<dyn Iterator<Item = Result<UnicodeLine, Error>> + 'a>;

    fn parse(&self) -> Result<Self::Output<'_>, Error> {
        let lines = self.mmap.as_slice().split(|&x| x == b'\n');

        let iter = lines
            .enumerate()
            .filter(|(_, text)| !text.is_empty())
            .map(|(idx, text)| {
                let line_no = idx + 1;
                let unicode_line = parse_line(text, &self.path, line_no)?;
                Ok(unicode_line)
            });

        Ok(Box::new(iter))
    }
}

fn parse_line(text: &[u8], path: &str, line: usize) -> Result<UnicodeLine, Error> {
    let fields = text.split(|x| *x == b';').collect::<Vec<_>>();
    if fields.len() != 15 {
        return Err(Error::broken_data(path, line));
    }

    let codepoint = parse_hex_u32(fields[0]);
    let combining_class = parse_u8(fields[3]);
    let decomposition = if fields[5].is_empty() || fields[5][0] == b'<' {
        Vec::new()
    } else {
        let split = fields[5].split(|x| *x == b' ');
        split.into_iter().map(parse_hex_u32).collect()
    };

    Ok(UnicodeLine {
        codepoint,
        combining_class,
        decomposition,
    })
}

pub struct UnicodeBinary {
    #[allow(unused)]
    path: String,
    mmap: utils::Mmap,
}

impl UnicodeBinary {
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
impl Binary for UnicodeBinary {
    fn raw(&self) -> Result<&[u8], Error> {
        Ok(self.mmap.as_slice())
    }
}
