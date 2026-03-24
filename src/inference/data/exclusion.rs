use super::{Binary, Error, Text, parse_hex_u32};
use crate::inference::utils;
use std::fs::File;

pub struct ExclusionText {
    #[allow(unused)]
    path: String,
    mmap: utils::Mmap,
}

impl ExclusionText {
    pub fn new(path: &str) -> Result<Self, Error> {
        let file = File::open(path).map_err(|err| Error::io(path, err))?;
        let mmap = utils::Mmap::new(&file).map_err(|err| Error::io(path, err))?;

        Ok(Self {
            path: path.to_string(),
            mmap,
        })
    }
}

impl Text for ExclusionText {
    type Output<'a> = Box<dyn Iterator<Item = Result<u32, Error>> + 'a>;

    fn parse(&self) -> Result<Self::Output<'_>, Error> {
        let lines = self.mmap.as_slice().split(|&x| x == b'\n');

        let iter = lines
            .filter(|text| !text.is_empty() && text[0] != b'#')
            .map(|text| Ok(parse_hex_u32(text)));

        Ok(Box::new(iter))
    }
}

#[allow(unused)]
pub struct ExclusionBinary {
    path: String,
    mmap: utils::Mmap,
}

impl ExclusionBinary {
    pub fn new(path: &str) -> Result<Self, Error> {
        let file = File::open(path).map_err(|err| Error::io(path, err))?;
        let mmap = utils::Mmap::new(&file).map_err(|err| Error::io(path, err))?;

        Ok(Self {
            path: path.to_string(),
            mmap,
        })
    }
}

impl Binary for ExclusionBinary {
    fn raw(&self) -> Result<&[u8], Error> {
        Ok(self.mmap.as_slice())
    }
}
