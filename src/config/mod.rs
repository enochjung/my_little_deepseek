mod composition_exclusion;
mod merge;
mod unicode;
mod vocab;
mod weight;

use crate::device::Cpu;
pub use composition_exclusion::CompositionExclusionFormat;
pub use merge::MergeFormat;
pub use unicode::UnicodeFormat;
pub use vocab::VocabFormat;
pub use weight::WeightFormat;

pub struct Configure {
    pub(crate) unicode_format: Option<UnicodeFormat>,
    pub(crate) composition_exclusion_format: Option<CompositionExclusionFormat>,
    pub(crate) merge_format: Option<MergeFormat>,
    pub(crate) vocab_format: Option<VocabFormat>,
    pub(crate) weight_format: Option<WeightFormat>,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) rms_norm_epsilon: f32,
    pub(crate) hidden_size: u32,
    pub(crate) intermediate_size: u32,
    pub(crate) rope_theta: f32,
    pub(crate) vocab_size: u32,
}

impl Configure {
    pub fn new() -> Self {
        Self {
            unicode_format: None,
            composition_exclusion_format: None,
            merge_format: None,
            vocab_format: None,
            weight_format: None,
            num_hidden_layers: 28,
            num_attention_heads: 12,
            num_key_value_heads: 2,
            rms_norm_epsilon: 1e-06,
            hidden_size: 1536,
            intermediate_size: 8960,
            rope_theta: 10000.0,
            vocab_size: 151936,
        }
    }

    pub fn unicode_format(mut self, value: UnicodeFormat) -> Self {
        self.unicode_format = Some(value);
        self
    }

    pub fn composition_exclusion_format(mut self, value: CompositionExclusionFormat) -> Self {
        self.composition_exclusion_format = Some(value);
        self
    }

    pub fn merge_format(mut self, value: MergeFormat) -> Self {
        self.merge_format = Some(value);
        self
    }

    pub fn vocab_format(mut self, value: VocabFormat) -> Self {
        self.vocab_format = Some(value);
        self
    }

    pub fn weight_format(mut self, value: WeightFormat) -> Self {
        self.weight_format = Some(value);
        self
    }

    pub fn num_hidden_layers(mut self, value: usize) -> Self {
        self.num_hidden_layers = value;
        self
    }

    pub fn num_attention_heads(mut self, value: usize) -> Self {
        self.num_attention_heads = value;
        self
    }

    pub fn num_key_value_heads(mut self, value: usize) -> Self {
        self.num_key_value_heads = value;
        self
    }

    pub fn rms_norm_epsilon(mut self, value: f32) -> Self {
        self.rms_norm_epsilon = value;
        self
    }

    pub fn hidden_size(mut self, value: u32) -> Self {
        self.hidden_size = value;
        self
    }

    pub fn intermediate_size(mut self, value: u32) -> Self {
        self.intermediate_size = value;
        self
    }

    pub fn rope_theta(mut self, value: f32) -> Self {
        self.rope_theta = value;
        self
    }

    pub fn vocab_size(mut self, value: u32) -> Self {
        self.vocab_size = value;
        self
    }
}

pub(crate) trait Format {
    type Output;
    type Parser: Fn(&Cpu) -> Box<dyn Iterator<Item = Self::Output> + '_>;

    fn read(&self) -> Result<(Cpu, Self::Parser), crate::Error>;
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
