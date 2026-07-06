use common::{BF16, Error};

use crate::Format;

use std::fs::File;
use std::marker::PhantomData;
use std::ops::Range;
use std::os::unix::fs::FileExt;

const HLEN_END_OFFSET: usize = 8;

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
    type Item = Result<WeightInfo<BF16>, Error>;
    type Parser = fn(&File) -> Box<dyn Iterator<Item = Self::Item> + '_>;

    fn open(&self) -> Result<(File, Self::Parser), Error> {
        match &self {
            WeightFormat::Safetensor { path } => {
                let file =
                    File::open(path).map_err(|err| Error::raw_os_error(err.raw_os_error()))?;
                Ok((file, parse_safetensor))
            }
        }
    }
}

fn parse_safetensor(file: &File) -> Box<dyn Iterator<Item = Result<WeightInfo<BF16>, Error>> + '_> {
    let (header, weight_start) = match read_header(file) {
        Ok(header) => header,
        Err(err) => return Box::new(std::iter::once(Err(err))),
    };

    Box::new(LineIter {
        header,
        weight_start,
        pos: 0,
    })
}

fn read_header(file: &File) -> Result<(Vec<u8>, usize), Error> {
    let mut hlen_bytes = [0; HLEN_END_OFFSET];
    let read = file
        .read_at(&mut hlen_bytes, 0)
        .map_err(|err| Error::raw_os_error(err.raw_os_error()))?;
    if read < hlen_bytes.len() {
        return Err(Error::broken_data(0));
    }

    let hlen = usize::from_le_bytes(hlen_bytes);

    let mut header = vec![0; hlen];
    let read = file
        .read_at(&mut header, HLEN_END_OFFSET as u64)
        .map_err(|err| Error::raw_os_error(err.raw_os_error()))?;
    if read < header.len() {
        return Err(Error::broken_data(0));
    }

    Ok((header, HLEN_END_OFFSET + hlen))
}

struct LineIter {
    header: Vec<u8>,
    weight_start: usize,
    pos: usize,
}

impl Iterator for LineIter {
    type Item = Result<WeightInfo<BF16>, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.header.len() < 2 {
            return None;
        }
        if self.pos == 0 {
            self.pos += 1;
        }
        if self.pos >= self.header.len() - 1 {
            return None;
        }

        let name = parse_string(&self.header, &mut self.pos).to_vec();
        self.pos += 1;

        let body = extract_value(&self.header, &mut self.pos).to_vec();
        if self.pos < self.header.len() && self.header[self.pos] == b',' {
            self.pos += 1;
        }

        if name == b"__metadata__" {
            return self.next();
        }

        Some(Ok(parse_tensor_info(&name, &body, self.weight_start)))
    }
}

fn parse_string<'a>(data: &'a [u8], pos: &mut usize) -> &'a [u8] {
    *pos += 1;
    let start = *pos;

    while *pos < data.len() && data[*pos] != b'"' {
        *pos += 1;
    }

    let end = *pos;
    *pos += 1;

    &data[start..end]
}

fn extract_value<'a>(data: &'a [u8], pos: &mut usize) -> &'a [u8] {
    let start = *pos;
    let end_token = match data[*pos] {
        b'"' => b'"',
        b'{' => b'}',
        _ => b']',
    };

    *pos += 1;
    while *pos < data.len() && data[*pos] != end_token {
        *pos += 1;
    }

    *pos += 1;
    &data[start..*pos]
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

fn parse_tensor_info(name: &[u8], body: &[u8], additional_offset: usize) -> WeightInfo<BF16> {
    let name = unsafe { std::str::from_utf8_unchecked(name) }.to_string();
    let mut pos = 1;

    parse_string(body, &mut pos);
    pos += 1;
    extract_value(body, &mut pos);
    if body[pos] == b',' {
        pos += 1;
    }

    parse_string(body, &mut pos);
    pos += 1;
    let section_shape = extract_value(body, &mut pos);
    let shape = parse_array(section_shape)
        .into_iter()
        .map(|x| x as u32)
        .collect();
    if body[pos] == b',' {
        pos += 1;
    }

    parse_string(body, &mut pos);
    pos += 1;
    let section_offsets = extract_value(body, &mut pos);
    let offset = parse_array(section_offsets);
    let offset = (offset[0] + additional_offset)..(offset[1] + additional_offset);

    WeightInfo {
        name,
        shape,
        offset,
        _phantom: PhantomData,
    }
}

#[cfg(test)]
mod tests {
    use common::BF16;

    use super::{WeightFormat, WeightInfo};
    use crate::Format;

    use std::collections::HashMap;
    use std::ops::Range;
    use std::sync::OnceLock;

    const WEIGHT_PATH: &'static str = "../../model/model.safetensors";
    const OFFSET: usize = 38621;

    static PRECOMPUTED_TENSOR_INFO: OnceLock<HashMap<String, WeightInfo<BF16>>> = OnceLock::new();

    fn get_tensor_info() -> &'static HashMap<String, WeightInfo<BF16>> {
        PRECOMPUTED_TENSOR_INFO.get_or_init(|| {
            let weight_format = WeightFormat::Safetensor {
                path: WEIGHT_PATH.to_string(),
            };
            let (file, parse) = weight_format.open().expect("Failed to read weight format");

            let iter = parse(&file);
            iter.map(|w| {
                let w = w.expect("Failed to parse weight info");
                (w.name.clone(), w)
            })
            .collect()
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
