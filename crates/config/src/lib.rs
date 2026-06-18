#![feature(stdarch_x86_avx512_bf16)]

mod composition_exclusion;
mod merge;
mod unicode;
mod vocab;
mod weight;

pub use composition_exclusion::CompositionExclusionFormat;
pub use merge::MergeFormat;
pub use unicode::UnicodeFormat;
pub use vocab::VocabFormat;
pub use weight::{WeightFormat, WeightInfo};

use backend_host::Mmap;
use core::MLTError;

/// Configuration blueprint for initializing the inference engine model.
///
/// `Configure` acts as a central registry that maps model artifacts and defines
/// the structural hyperparameters of the neural network. It serves as the primary
/// input for the `Model::new` constructor, decoupling the model's architectural
/// definition from the physical loading of weights.
///
/// # Examples
///
/// ```no_run
/// let config = config::Configure::new()
///     .unicode_format(config::UnicodeFormat::UnicodeCharacterDatabase {
///         path: "path/to/UnicodeData.txt".to_string(),
///     })
///     .composition_exclusion_format(
///         config::CompositionExclusionFormat::UnicodeCharacterDatabase {
///             path: "path/to/CompositionExclusions".to_string(),
///         },
///     )
///     .merge_format(config::MergeFormat::HuggingFace {
///         path: "path/to/merges.json".to_string(),
///     })
///     .vocab_format(config::VocabFormat::HuggingFace {
///         path: "path/to/vocab.json".to_string(),
///     })
///     .weight_format(config::WeightFormat::Safetensor {
///         path: "path/to/model.safetensors".to_string(),
///     })
///     .num_hidden_layers(28)
///     .hidden_size(1536);
/// ```
pub struct Configure {
    pub unicode_format: Option<UnicodeFormat>,
    pub composition_exclusion_format: Option<CompositionExclusionFormat>,
    pub merge_format: Option<MergeFormat>,
    pub vocab_format: Option<VocabFormat>,
    pub weight_format: Option<WeightFormat>,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_epsilon: f32,
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub rope_theta: f32,
    pub vocab_size: u32,
}

impl Default for Configure {
    fn default() -> Self {
        Self::new()
    }
}

impl Configure {
    /// Creates a new, default configuration for the model.
    ///
    /// The default values are initialized to provide a baseline for common model architectures.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use config::Configure;
    ///
    /// let config = Configure::new();
    /// ```
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

    /// Sets the format and path for the Unicode character database configuration.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use config::{Configure, UnicodeFormat};
    ///
    /// let config = Configure::new().unicode_format(
    ///     UnicodeFormat::UnicodeCharacterDatabase { path: "path/to/unicode.txt".to_string() }
    /// );
    /// ```
    pub fn unicode_format(mut self, value: UnicodeFormat) -> Self {
        self.unicode_format = Some(value);
        self
    }

    /// Sets the format and path for the composition exclusion configuration.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use config::{Configure, CompositionExclusionFormat};
    ///
    /// let config = Configure::new().composition_exclusion_format(
    ///     CompositionExclusionFormat::UnicodeCharacterDatabase { path: "path/to/exclusion.txt".to_string() }
    /// );
    /// ```
    pub fn composition_exclusion_format(mut self, value: CompositionExclusionFormat) -> Self {
        self.composition_exclusion_format = Some(value);
        self
    }

    /// Sets the format and path for the merge operations configuration.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use config::{Configure, MergeFormat};
    ///
    /// let config = Configure::new().merge_format(
    ///     MergeFormat::HuggingFace { path: "path/to/merges.json".to_string() }
    /// );
    /// ```
    pub fn merge_format(mut self, value: MergeFormat) -> Self {
        self.merge_format = Some(value);
        self
    }

    /// Sets the format and path for the vocabulary configuration.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use config::{Configure, VocabFormat};
    ///
    /// let config = Configure::new().vocab_format(
    ///     VocabFormat::HuggingFace { path: "path/to/vocab.json".to_string() }
    /// );
    /// ```
    pub fn vocab_format(mut self, value: VocabFormat) -> Self {
        self.vocab_format = Some(value);
        self
    }

    /// Sets the format and path for the model weights configuration.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use config::{Configure, WeightFormat};
    ///
    /// let config = Configure::new().weight_format(
    ///     WeightFormat::Safetensor { path: "path/to/model.safetensors".to_string() }
    /// );
    /// ```
    pub fn weight_format(mut self, value: WeightFormat) -> Self {
        self.weight_format = Some(value);
        self
    }

    /// Sets the number of hidden layers in the transformer model.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use config::Configure;
    ///
    /// let config = Configure::new().num_hidden_layers(32);
    /// ```
    pub fn num_hidden_layers(mut self, value: usize) -> Self {
        self.num_hidden_layers = value;
        self
    }

    /// Sets the number of attention heads.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use config::Configure;
    ///
    /// let config = Configure::new().num_attention_heads(16);
    /// ```
    pub fn num_attention_heads(mut self, value: usize) -> Self {
        self.num_attention_heads = value;
        self
    }

    /// Sets the number of key-value attention heads for grouped-query attention.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use config::Configure;
    ///
    /// let config = Configure::new().num_key_value_heads(4);
    /// ```
    pub fn num_key_value_heads(mut self, value: usize) -> Self {
        self.num_key_value_heads = value;
        self
    }

    /// Sets the epsilon value used for RMS normalization stability.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use config::Configure;
    ///
    /// let config = Configure::new().rms_norm_epsilon(1e-5);
    /// ```
    pub fn rms_norm_epsilon(mut self, value: f32) -> Self {
        self.rms_norm_epsilon = value;
        self
    }

    /// Sets the hidden layer size (dimension) of the model.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use config::Configure;
    ///
    /// let config = Configure::new().hidden_size(2048);
    /// ```
    pub fn hidden_size(mut self, value: u32) -> Self {
        self.hidden_size = value;
        self
    }

    /// Sets the intermediate size for the feed-forward networks.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use config::Configure;
    ///
    /// let config = Configure::new().intermediate_size(8192);
    /// ```
    pub fn intermediate_size(mut self, value: u32) -> Self {
        self.intermediate_size = value;
        self
    }

    /// Sets the base theta value for Rotary Positional Embeddings (RoPE).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use config::Configure;
    ///
    /// let config = Configure::new().rope_theta(5000.0);
    /// ```
    pub fn rope_theta(mut self, value: f32) -> Self {
        self.rope_theta = value;
        self
    }

    /// Sets the total vocabulary size for tokenization.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use config::Configure;
    ///
    /// let config = Configure::new().vocab_size(151936);
    /// ```
    pub fn vocab_size(mut self, value: u32) -> Self {
        self.vocab_size = value;
        self
    }
}

pub trait Format {
    type Output;
    type Parser: Fn(&Mmap<u8>) -> Box<dyn Iterator<Item = Self::Output> + '_>;

    fn read(&self) -> Result<(Mmap<u8>, Self::Parser), MLTError>;
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
