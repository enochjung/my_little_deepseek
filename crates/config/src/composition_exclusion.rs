use common::Error;

use super::{Format, parse_hex_u32};

use std::fs::File;
use std::io::{BufRead, BufReader};

pub enum CompositionExclusionFormat {
    UnicodeCharacterDatabase { path: String },
}

impl Format for CompositionExclusionFormat {
    type Item = u32;
    type Parser = fn(&File) -> Box<dyn Iterator<Item = Self::Item> + '_>;

    fn open(&self) -> Result<(File, Self::Parser), Error> {
        match &self {
            CompositionExclusionFormat::UnicodeCharacterDatabase { path } => {
                let file =
                    File::open(path).map_err(|err| Error::raw_os_error(err.raw_os_error()))?;
                Ok((file, parse_ucd))
            }
        }
    }
}

fn parse_ucd(file: &File) -> Box<dyn Iterator<Item = u32> + '_> {
    Box::new(LineIter {
        reader: BufReader::new(file),
        buf: Vec::with_capacity(256),
    })
}

struct LineIter<'a> {
    reader: BufReader<&'a File>,
    buf: Vec<u8>,
}

impl Iterator for LineIter<'_> {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            self.buf.clear();

            if self.reader.read_until(b'\n', &mut self.buf).ok()? == 0 {
                return None;
            }

            if self.buf.last() == Some(&b'\n') {
                self.buf.pop();
            }
            if self.buf.last() == Some(&b'\r') {
                self.buf.pop();
            }
            if self.buf.is_empty() || self.buf[0] == b'#' {
                continue;
            }

            return Some(parse_hex_u32(&self.buf));
        }
    }
}
