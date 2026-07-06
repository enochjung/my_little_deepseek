use common::Error;

use super::{Format, parse_string_with_escape_sequence};

use std::fs::File;
use std::io::{BufRead, BufReader};

pub enum MergeFormat {
    HuggingFace { path: String },
}

impl Format for MergeFormat {
    type Item = Result<(String, String), Error>;
    type Parser = fn(&File) -> Box<dyn Iterator<Item = Self::Item> + '_>;

    fn open(&self) -> Result<(File, Self::Parser), Error> {
        match &self {
            MergeFormat::HuggingFace { path } => {
                let file =
                    File::open(path).map_err(|err| Error::raw_os_error(err.raw_os_error()))?;
                Ok((file, parse_huggingface))
            }
        }
    }
}

fn parse_huggingface(
    file: &File,
) -> Box<dyn Iterator<Item = Result<(String, String), Error>> + '_> {
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
    type Item = Result<(String, String), Error>;

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
            if self.buf.is_empty() || self.buf[0] != b' ' {
                continue;
            }

            return Some(parse_line(&self.buf, self.line_no));
        }
    }
}

fn parse_line(text: &[u8], line: usize) -> Result<(String, String), Error> {
    if text.len() < 7 || text[0] != b' ' || text[1] != b' ' || text[2] != b'"' {
        return Err(Error::broken_data(line));
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
                return Err(Error::broken_data(line));
            }
        }
        cqi += 1;
    }

    if text.len() < cqi + 1 || text[cqi] != b'"' || si == 0 {
        return Err(Error::broken_data(line));
    }

    let left = parse_string_with_escape_sequence(&text[3..si]);
    let right = parse_string_with_escape_sequence(&text[si + 1..cqi]);

    Ok((left, right))
}
