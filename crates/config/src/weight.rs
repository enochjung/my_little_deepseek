use backend_host::Mmap;
use core::MLTError;

use crate::Format;

use std::arch::x86_64::bf16;
use std::fs::File;
use std::marker::PhantomData;
use std::ops::Range;

pub struct WeightInfo<T> {
    pub name: String,
    pub shape: Vec<u32>,
    pub offset: Range<usize>,
    _phantom: PhantomData<T>,
}

pub enum WeightFormat {
    Safetensor { path: String },
}

impl Format for WeightFormat {
    type Output = WeightInfo<bf16>;
    type Parser = fn(&Mmap<u8>) -> Box<dyn Iterator<Item = Self::Output> + '_>;

    fn read(&self) -> Result<(Mmap<u8>, Self::Parser), MLTError> {
        match &self {
            WeightFormat::Safetensor { path } => {
                let file = File::open(path).map_err(MLTError::io)?;
                let mem = Mmap::try_from(file)?;
                Ok((mem, safetensor::parse))
            }
        }
    }
}

mod safetensor {
    use backend_host::Mmap;

    use super::WeightInfo;

    use std::arch::x86_64::bf16;
    use std::marker::PhantomData;

    const HLEN_END_OFFSET: usize = 8;

    pub fn parse(file: &Mmap<u8>) -> Box<dyn Iterator<Item = WeightInfo<bf16>> + '_> {
        let raw = file.as_slice();

        let (header_start, weight_start) = section_offsets(raw);

        let header = &raw[header_start..weight_start];
        let iter = SectionIter::new(header);

        Box::new(
            iter.filter(is_not_metadata)
                .map(move |section| parse_tensor_info(section, weight_start)),
        )
    }

    fn section_offsets(raw: &[u8]) -> (usize, usize) {
        let mut hlen_bytes = [0; HLEN_END_OFFSET];
        hlen_bytes.copy_from_slice(&raw[..HLEN_END_OFFSET]);
        let hlen = usize::from_le_bytes(hlen_bytes);

        let header_start = HLEN_END_OFFSET;
        let header_end = header_start + hlen;

        (header_start, header_end)
    }

    struct SectionIter<'a> {
        data: &'a [u8],
        pos: usize,
    }

    struct Section<'a> {
        name: &'a [u8],
        body: &'a [u8],
    }

    impl<'a> SectionIter<'a> {
        pub fn new(data: &'a [u8]) -> Self {
            let data = &data[1..data.len() - 1];
            Self { data, pos: 0 }
        }

        pub fn parse_string(&mut self) -> &'a [u8] {
            self.pos += 1;
            let start = self.pos;

            while self.pos < self.data.len() && self.data[self.pos] != b'"' {
                self.pos += 1;
            }

            let end = self.pos;
            self.pos += 1;

            &self.data[start..end]
        }

        pub fn extract_value(&mut self) -> &'a [u8] {
            let start = self.pos;
            let end_token = match self.data[self.pos] {
                b'"' => b'"',
                b'{' => b'}',
                _ => b']',
            };

            self.pos += 1;
            while self.pos < self.data.len() && self.data[self.pos] != end_token {
                self.pos += 1;
            }

            self.pos += 1;
            &self.data[start..self.pos]
        }
    }

    impl<'a> Iterator for SectionIter<'a> {
        type Item = Section<'a>;

        fn next(&mut self) -> Option<Self::Item> {
            if self.pos >= self.data.len() {
                return None;
            }

            let key = self.parse_string();
            self.pos += 1;

            let value = self.extract_value();
            if self.pos < self.data.len() && self.data[self.pos] == b',' {
                self.pos += 1;
            }

            Some(Self::Item {
                name: key,
                body: value,
            })
        }
    }

    fn is_not_metadata(section: &Section) -> bool {
        section.name != b"__metadata__"
    }

    fn parse_array(text: &[u8]) -> Vec<usize> {
        let mut res = Vec::new();
        let mut num = 0;

        for &byte in &text[1..text.len() - 1] {
            match byte {
                b',' => {
                    res.push(num);
                    num = 0;
                }
                _ => {
                    num *= 10;
                    num += (byte - b'0') as usize;
                }
            }
        }
        res.push(num);

        res
    }

    fn parse_tensor_info(section: Section, additional_offset: usize) -> WeightInfo<bf16> {
        let name = unsafe { std::str::from_utf8_unchecked(section.name) }.to_string();
        let mut parser = SectionIter::new(section.body);

        parser.next();

        let section_shape = parser.next().unwrap();
        let shape = parse_array(section_shape.body)
            .into_iter()
            .map(|x| x as u32)
            .collect();

        let section_offsets = parser.next().unwrap();
        let offset = parse_array(section_offsets.body);
        let offset = (offset[0] + additional_offset)..(offset[1] + additional_offset);

        WeightInfo {
            name,
            shape,
            offset,
            _phantom: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{WeightFormat, WeightInfo};
    use crate::Format;

    use std::arch::x86_64::bf16;
    use std::collections::HashMap;
    use std::ops::Range;
    use std::sync::OnceLock;

    const WEIGHT_PATH: &'static str = "../../model/model.safetensors";
    const OFFSET: usize = 38621;

    static PRECOMPUTED_TENSOR_INFO: OnceLock<HashMap<String, WeightInfo<bf16>>> = OnceLock::new();

    fn get_tensor_info() -> &'static HashMap<String, WeightInfo<bf16>> {
        PRECOMPUTED_TENSOR_INFO.get_or_init(|| {
            let weight_format = WeightFormat::Safetensor {
                path: WEIGHT_PATH.to_string(),
            };
            let (file, parse) = weight_format.read().expect("Failed to read weight format");

            let iter = parse(&file);
            iter.map(|w| (w.name.clone(), w)).collect()
        })
    }

    fn assert(tensor_name: &str, expected_shape: &[u32], expected_offset: Range<usize>) {
        let tensor_info = get_tensor_info();

        if let Some(info) = tensor_info.get(tensor_name) {
            assert_eq!(
                expected_shape, info.shape,
                "expected: {:?}, actual: {:?}",
                expected_shape, info.shape
            );
            assert_eq!(
                expected_offset, info.offset,
                "expected: {:?}, actual: {:?}",
                expected_offset, info.offset,
            );
            return;
        }

        panic!("tensor '{}' missing", tensor_name);
    }

    #[test]
    fn text_embed_tokens_weight() {
        assert(
            "model.embed_tokens.weight",
            &[151936, 1536],
            0 + OFFSET..466747392 + OFFSET,
        );
    }

    #[test]
    fn text_q_proj_bias() {
        assert(
            "model.layers.0.self_attn.q_proj.bias",
            &[1536],
            466747392 + OFFSET..466750464 + OFFSET,
        );
    }

    #[test]
    fn text_k_proj_weight() {
        assert(
            "model.layers.1.self_attn.k_proj.weight",
            &[256, 1536],
            565065728 + OFFSET..565852160 + OFFSET,
        );
    }

    #[test]
    fn text_v_proj_weight() {
        assert(
            "model.layers.2.self_attn.v_proj.weight",
            &[256, 1536],
            659447808 + OFFSET..660234240 + OFFSET,
        );
    }

    #[test]
    fn text_o_proj_weight() {
        assert(
            "model.layers.3.self_attn.o_proj.weight",
            &[1536, 1536],
            753829888 + OFFSET..758548480 + OFFSET,
        );
    }

    #[test]
    fn text_gate_proj_weight() {
        assert(
            "model.layers.4.mlp.gate_proj.weight",
            &[8960, 1536],
            852144128 + OFFSET..879669248 + OFFSET,
        );
    }

    #[test]
    fn text_up_proj_weight() {
        assert(
            "model.layers.5.mlp.up_proj.weight",
            &[8960, 1536],
            973264896 + OFFSET..1000790016 + OFFSET,
        );
    }

    #[test]
    fn text_down_proj_weight() {
        assert(
            "model.layers.6.mlp.down_proj.weight",
            &[1536, 8960],
            1094385664 + OFFSET..1121910784 + OFFSET,
        );
    }

    #[test]
    fn text_input_layernorm_weight() {
        assert(
            "model.layers.7.input_layernorm.weight",
            &[1536],
            1215506432 + OFFSET..1215509504 + OFFSET,
        );
    }

    #[test]
    fn text_post_attention_layernorm_weight() {
        assert(
            "model.layers.27.post_attention_layernorm.weight",
            &[1536],
            3087422464 + OFFSET..3087425536 + OFFSET,
        );
    }

    #[test]
    fn text_norm_weight() {
        assert(
            "model.norm.weight",
            &[1536],
            3087425536 + OFFSET..3087428608 + OFFSET,
        );
    }

    #[test]
    fn text_lm_head_weight() {
        assert(
            "lm_head.weight",
            &[151936, 1536],
            3087428608 + OFFSET..3554176000 + OFFSET,
        );
    }
}
