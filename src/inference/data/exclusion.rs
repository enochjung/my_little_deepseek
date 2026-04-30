use super::{Binary, Error, Text, parse_hex_u32};
use crate::inference::utils;
use std::fs::File;

pub struct ExclusionText {
    path: String,
    data: utils::Mmap,
}

impl ExclusionText {
    pub fn new(path: &str) -> Result<Self, Error> {
        let file = File::open(path).map_err(|err| Error::io(path, err))?;
        let data = utils::Mmap::from(&file);

        Ok(Self {
            path: path.to_string(),
            data,
        })
    }
}

impl Text for ExclusionText {
    type Output<'a> = Box<dyn Iterator<Item = Result<u32, Error>> + 'a>;

    fn parse(&self) -> Result<Self::Output<'_>, Error> {
        let lines = self.data.as_slice().split(|&x| x == b'\n');

        let iter = lines
            .filter(|text| !text.is_empty() && text[0] != b'#')
            .map(|text| Ok(parse_hex_u32(text)));

        Ok(Box::new(iter))
    }
}

pub struct ExclusionBinary {
    path: String,
    data: utils::Mmap,
}

impl ExclusionBinary {
    pub fn new(path: &str) -> Result<Self, Error> {
        let file = File::open(path).map_err(|err| Error::io(path, err))?;
        let data = utils::Mmap::from(&file);

        Ok(Self {
            path: path.to_string(),
            data,
        })
    }
}

impl Binary for ExclusionBinary {
    fn raw(&self) -> Result<&[u8], Error> {
        Ok(self.data.as_slice())
    }
}
