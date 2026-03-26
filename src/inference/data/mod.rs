mod exclusion;
mod merge;
mod unicode;
mod vocab;
mod weight;

pub use exclusion::{ExclusionBinary, ExclusionText};
pub use merge::{MergeBinary, MergeText};
pub use unicode::{UnicodeBinary, UnicodeText};
pub use vocab::{VocabBinary, VocabText};
pub use weight::{WeightBinary, WeightText};

use super::Error;
use std::path::Path;

pub struct ModelData {
    pub unicode_text: Option<UnicodeText>,
    pub unicode_binary: Option<UnicodeBinary>,
    pub exclusion_text: Option<ExclusionText>,
    pub exclusion_binary: Option<ExclusionBinary>,
    pub merge_text: Option<MergeText>,
    pub merge_binary: Option<MergeBinary>,
    pub vocab_text: Option<VocabText>,
    pub vocab_binary: Option<VocabBinary>,
    #[allow(unused)]
    pub weight_text: Option<WeightText>,
    #[allow(unused)]
    pub weight_binary: Option<WeightBinary>,
}

pub trait Text {
    type Output<'a>
    where
        Self: 'a;

    fn parse(&self) -> Result<Self::Output<'_>, Error>;
}

pub trait Binary {
    #[allow(unused)]
    fn raw(&self) -> Result<&[u8], Error>;
}

impl ModelData {
    pub fn new(
        unicode_path: &str,
        exclusion_path: &str,
        merge_path: &str,
        vocab_path: &str,
        weight_path: &str,
    ) -> Result<Self, Error> {
        let (unicode_text, unicode_binary) = match Path::new(unicode_path)
            .extension()
            .and_then(|ext| ext.to_str())
        {
            Some("txt") => (Some(UnicodeText::new(unicode_path)?), None),
            Some("bin") => (None, Some(UnicodeBinary::new(unicode_path)?)),
            Some("none") => (None, None),
            _ => return Err(Error::unknown_format(unicode_path)),
        };

        let (exclusion_text, exclusion_binary) = match Path::new(exclusion_path)
            .extension()
            .and_then(|ext| ext.to_str())
        {
            Some("txt") => (Some(ExclusionText::new(exclusion_path)?), None),
            Some("bin") => (None, Some(ExclusionBinary::new(exclusion_path)?)),
            Some("none") => (None, None),
            _ => return Err(Error::unknown_format(exclusion_path)),
        };

        let (merge_text, merge_binary) = match Path::new(merge_path)
            .extension()
            .and_then(|ext| ext.to_str())
        {
            Some("json") => (Some(MergeText::new(merge_path)?), None),
            Some("bin") => (None, Some(MergeBinary::new(merge_path)?)),
            Some("none") => (None, None),
            _ => return Err(Error::unknown_format(merge_path)),
        };

        let (vocab_text, vocab_binary) = match Path::new(vocab_path)
            .extension()
            .and_then(|ext| ext.to_str())
        {
            Some("json") => (Some(VocabText::new(vocab_path)?), None),
            Some("bin") => (None, Some(VocabBinary::new(vocab_path)?)),
            Some("none") => (None, None),
            _ => return Err(Error::unknown_format(vocab_path)),
        };

        let (weight_text, weight_binary) = match Path::new(weight_path)
            .extension()
            .and_then(|ext| ext.to_str())
        {
            Some("safetensors") => (Some(WeightText::new(weight_path)?), None),
            Some("bin") => (None, Some(WeightBinary::new(weight_path)?)),
            Some("none") => (None, None),
            _ => return Err(Error::unknown_format(weight_path)),
        };

        Ok(Self {
            unicode_text,
            unicode_binary,
            exclusion_text,
            exclusion_binary,
            merge_text,
            merge_binary,
            vocab_text,
            vocab_binary,
            weight_text,
            weight_binary,
        })
    }
}

fn parse_hex_u32(text: &[u8]) -> u32 {
    let mut value: u32 = 0;
    for &b in text {
        let digit = match b {
            b'0'..=b'9' => (b - b'0') as u32,
            b'A'..=b'F' => (b - b'A' + 10) as u32,
            b'a'..=b'f' => (b - b'a' + 10) as u32,
            _ => break,
        };
        value = value * 16 + digit;
    }
    value
}

fn parse_u8(text: &[u8]) -> u8 {
    let mut value: u8 = 0;
    for &b in text {
        let digit = match b {
            b'0'..=b'9' => b - b'0',
            _ => break,
        };
        value = value * 10 + digit;
    }
    value
}

fn parse_u32(text: &[u8]) -> u32 {
    let mut value: u32 = 0;
    for &b in text {
        let digit = match b {
            b'0'..=b'9' => (b - b'0') as u32,
            _ => break,
        };
        value = value * 10 + digit;
    }
    value
}

fn parse_string_with_escape_sequence(text: &[u8]) -> String {
    let mut out = Vec::with_capacity(text.len());
    let mut i = 0;

    while i < text.len() {
        if text[i] != b'\\' {
            out.push(text[i]);
            i += 1;
            continue;
        }

        if i + 1 >= text.len() {
            out.push(b'\\');
            break;
        }

        match text[i + 1] {
            b'"' => {
                out.push(b'"');
                i += 2;
            }
            b'\\' => {
                out.push(b'\\');
                i += 2;
            }
            b'/' => {
                out.push(b'/');
                i += 2;
            }
            b'b' => {
                out.push(0x08);
                i += 2;
            }
            b'f' => {
                out.push(0x0C);
                i += 2;
            }
            b'n' => {
                out.push(b'\n');
                i += 2;
            }
            b'r' => {
                out.push(b'\r');
                i += 2;
            }
            b't' => {
                out.push(b'\t');
                i += 2;
            }
            b'u' => {
                if i + 6 <= text.len() && text[i + 2..i + 6].iter().all(|&x| x.is_ascii_hexdigit())
                {
                    let code = parse_hex_u32(&text[i + 2..i + 6]);
                    if let Some(ch) = char::from_u32(code) {
                        let mut buf = [0u8; 4];
                        let encoded = ch.encode_utf8(&mut buf);
                        out.extend_from_slice(encoded.as_bytes());
                    }
                    i += 6;
                } else {
                    out.push(b'\\');
                    i += 1;
                }
            }
            _ => {
                out.push(b'\\');
                out.push(text[i + 1]);
                i += 2;
            }
        }
    }

    String::from_utf8(out).unwrap_or_else(|_| String::from_utf8_lossy(text).into_owned())
}
