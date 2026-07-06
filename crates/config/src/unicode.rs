use common::Error;

use super::{Format, parse_hex_u32, parse_u8};

use std::fs::File;
use std::io::{BufRead, BufReader};

pub enum UnicodeFormat {
    UnicodeCharacterDatabase { path: String },
}

pub struct UCDLine {
    pub codepoint: u32,
    pub combining_class: u8,
    pub decomposition: Vec<u32>,
}

impl Format for UnicodeFormat {
    type Item = Result<UCDLine, Error>;
    type Parser = fn(&File) -> Box<dyn Iterator<Item = Self::Item> + '_>;

    fn open(&self) -> Result<(File, Self::Parser), Error> {
        match &self {
            UnicodeFormat::UnicodeCharacterDatabase { path } => {
                let file =
                    File::open(path).map_err(|err| Error::raw_os_error(err.raw_os_error()))?;
                Ok((file, parse_ucd))
            }
        }
    }
}

fn parse_ucd(file: &File) -> Box<dyn Iterator<Item = Result<UCDLine, Error>> + '_> {
    Box::new(LineIter {
        reader: BufReader::new(file),
        buf: Vec::with_capacity(256),
        line_no: 0,
    })
}

struct LineIter<'a> {
    reader: BufReader<&'a File>,
    buf: Vec<u8>,
    line_no: usize,
}

impl Iterator for LineIter<'_> {
    type Item = Result<UCDLine, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            self.buf.clear();

            let read = match self.reader.read_until(b'\n', &mut self.buf) {
                Ok(read) => read,
                Err(_) => return Some(Err(Error::broken_data(self.line_no + 1))),
            };

            if read == 0 {
                return None;
            }

            self.line_no += 1;

            if self.buf.last() == Some(&b'\n') {
                self.buf.pop();
            }
            if self.buf.last() == Some(&b'\r') {
                self.buf.pop();
            }
            if self.buf.is_empty() {
                continue;
            }

            return Some(parse_ucd_line(&self.buf, self.line_no));
        }
    }
}

fn parse_ucd_line(text: &[u8], line_no: usize) -> Result<UCDLine, Error> {
    let fields = text.split(|x| *x == b';').collect::<Vec<_>>();
    if fields.len() != 15 {
        return Err(Error::broken_data(line_no));
    }

    let codepoint = parse_hex_u32(fields[0]);
    let combining_class = parse_u8(fields[3]);
    let decomposition = if fields[5].is_empty() || fields[5][0] == b'<' {
        Vec::new()
    } else {
        fields[5].split(|x| *x == b' ').map(parse_hex_u32).collect()
    };

    Ok(UCDLine {
        codepoint,
        combining_class,
        decomposition,
    })
}
