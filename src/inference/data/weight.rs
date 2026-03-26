use super::{Binary, Error, Text};
use crate::inference::utils;
use std::fs::File;
use std::{collections::HashMap, ops::Range};

const NUM_LAYER: usize = 28;

#[allow(unused)]
pub struct TensorInfo {
    pub shape: Vec<u32>,
    pub offset: Range<u64>,
}

#[allow(unused)]
pub struct LayerInfo {
    pub input_layernorm_weight: TensorInfo,
    pub q_proj_bias: TensorInfo,
    pub q_proj_weight: TensorInfo,
    pub k_proj_bias: TensorInfo,
    pub k_proj_weight: TensorInfo,
    pub v_proj_bias: TensorInfo,
    pub v_proj_weight: TensorInfo,
    pub o_proj_weight: TensorInfo,
    pub post_attention_layernorm_weight: TensorInfo,
    pub gate_proj_weight: TensorInfo,
    pub up_proj_weight: TensorInfo,
    pub down_proj_weight: TensorInfo,
}

#[allow(unused)]
pub struct WeightInfo {
    pub embed_tokens_weight: TensorInfo,
    pub layers: [LayerInfo; NUM_LAYER],
    pub norm_weight: TensorInfo,
    pub lm_head_weight: TensorInfo,
}

pub struct WeightText {
    path: String,
    mmap: utils::Mmap,
}

impl WeightText {
    pub fn new(path: &str) -> Result<Self, Error> {
        let file = File::open(path).map_err(|err| Error::io(path, err))?;
        let mmap = utils::Mmap::new(&file).map_err(|err| Error::io(path, err))?;

        Ok(Self {
            path: path.to_string(),
            mmap,
        })
    }
}

impl Text for WeightText {
    type Output<'a> = (WeightInfo, &'a [u8]);

    fn parse(&self) -> Result<Self::Output<'_>, Error> {
        let raw = self.mmap.as_slice();
        let (header_range, payload_range) = split_sections(raw, &self.path)?;

        let header = &raw[header_range];
        let tensors = parse_header(header, &self.path)?;
        let weight_info = WeightInfo::new(tensors)?;

        let payload = &raw[payload_range.start..payload_range.end];

        Ok((weight_info, payload))
    }
}

fn split_sections(raw: &[u8], path: &str) -> Result<(Range<usize>, Range<usize>), Error> {
    if raw.len() < 8 {
        return Err(Error::broken_data(path, 0));
    }

    let mut header_len_bytes = [0u8; 8];
    header_len_bytes.copy_from_slice(&raw[..8]);
    let header_len = u64::from_le_bytes(header_len_bytes);

    let header_start = 8usize;
    let header_end = header_start
        .checked_add(header_len as usize)
        .ok_or_else(|| Error::broken_data(path, 0))?;

    if header_end >= raw.len() {
        return Err(Error::broken_data(path, 0));
    }

    let header = &raw[header_start..header_end];
    if header.first() != Some(&b'{') || header.last() != Some(&b'}') {
        return Err(Error::broken_data(path, 0));
    }

    Ok((header_start..header_end, header_end..raw.len()))
}

fn parse_header(header: &[u8], path: &str) -> Result<HashMap<String, TensorInfo>, Error> {
    HeaderParser::new(header, path).parse_tensor_map()
}

fn take_tensor(tensors: &mut HashMap<String, TensorInfo>, key: &str) -> Result<TensorInfo, Error> {
    tensors
        .remove(key)
        .ok_or_else(|| Error::data_not_provided(key))
}

impl WeightInfo {
    fn new(mut tensors: HashMap<String, TensorInfo>) -> Result<Self, Error> {
        let embed_tokens_weight = take_tensor(&mut tensors, "model.embed_tokens.weight")?;
        let norm_weight = take_tensor(&mut tensors, "model.norm.weight")?;
        let lm_head_weight = take_tensor(&mut tensors, "lm_head.weight")?;

        let mut layers_vec = Vec::with_capacity(NUM_LAYER);
        for layer_idx in 0..NUM_LAYER {
            layers_vec.push(LayerInfo {
                input_layernorm_weight: take_tensor(
                    &mut tensors,
                    &format!("model.layers.{layer_idx}.input_layernorm.weight"),
                )?,
                q_proj_bias: take_tensor(
                    &mut tensors,
                    &format!("model.layers.{layer_idx}.self_attn.q_proj.bias"),
                )?,
                q_proj_weight: take_tensor(
                    &mut tensors,
                    &format!("model.layers.{layer_idx}.self_attn.q_proj.weight"),
                )?,
                k_proj_bias: take_tensor(
                    &mut tensors,
                    &format!("model.layers.{layer_idx}.self_attn.k_proj.bias"),
                )?,
                k_proj_weight: take_tensor(
                    &mut tensors,
                    &format!("model.layers.{layer_idx}.self_attn.k_proj.weight"),
                )?,
                v_proj_bias: take_tensor(
                    &mut tensors,
                    &format!("model.layers.{layer_idx}.self_attn.v_proj.bias"),
                )?,
                v_proj_weight: take_tensor(
                    &mut tensors,
                    &format!("model.layers.{layer_idx}.self_attn.v_proj.weight"),
                )?,
                o_proj_weight: take_tensor(
                    &mut tensors,
                    &format!("model.layers.{layer_idx}.self_attn.o_proj.weight"),
                )?,
                post_attention_layernorm_weight: take_tensor(
                    &mut tensors,
                    &format!("model.layers.{layer_idx}.post_attention_layernorm.weight"),
                )?,
                gate_proj_weight: take_tensor(
                    &mut tensors,
                    &format!("model.layers.{layer_idx}.mlp.gate_proj.weight"),
                )?,
                up_proj_weight: take_tensor(
                    &mut tensors,
                    &format!("model.layers.{layer_idx}.mlp.up_proj.weight"),
                )?,
                down_proj_weight: take_tensor(
                    &mut tensors,
                    &format!("model.layers.{layer_idx}.mlp.down_proj.weight"),
                )?,
            });
        }
        let layers: [LayerInfo; NUM_LAYER] = layers_vec
            .try_into()
            .map_err(|_| Error::broken_data("weight", 0))?;

        Ok(Self {
            embed_tokens_weight,
            layers,
            norm_weight,
            lm_head_weight,
        })
    }
}

pub struct WeightBinary {
    #[allow(unused)]
    path: String,
    mmap: utils::Mmap,
}

impl WeightBinary {
    pub fn new(path: &str) -> Result<Self, Error> {
        let file = File::open(path).map_err(|err| Error::io(path, err))?;
        let mmap = utils::Mmap::new(&file).map_err(|err| Error::io(path, err))?;

        Ok(Self {
            path: path.to_string(),
            mmap,
        })
    }
}

impl Binary for WeightBinary {
    #[allow(unused)]
    fn raw(&self) -> Result<&[u8], Error> {
        Ok(self.mmap.as_slice())
    }
}

struct HeaderParser<'a> {
    text: &'a [u8],
    idx: usize,
    path: &'a str,
}

impl<'a> HeaderParser<'a> {
    fn new(text: &'a [u8], path: &'a str) -> Self {
        Self { text, idx: 0, path }
    }

    fn parse_tensor_map(mut self) -> Result<HashMap<String, TensorInfo>, Error> {
        let mut tensors = HashMap::new();

        self.skip_ws();
        self.expect_byte(b'{')?;
        self.skip_ws();

        if self.try_byte(b'}') {
            return Ok(tensors);
        }

        loop {
            let name = self.parse_string()?;
            self.skip_ws();
            self.expect_byte(b':')?;
            self.skip_ws();

            if name == "__metadata__" {
                self.skip_json_value()?;
            } else {
                let info = self.parse_tensor_info()?;
                tensors.insert(name, info);
            }

            self.skip_ws();
            if self.try_byte(b',') {
                self.skip_ws();
                continue;
            }

            self.expect_byte(b'}')?;
            self.skip_ws();
            if self.idx != self.text.len() {
                return Err(Error::broken_data(self.path, 0));
            }
            break;
        }

        Ok(tensors)
    }

    fn parse_tensor_info(&mut self) -> Result<TensorInfo, Error> {
        self.expect_byte(b'{')?;
        self.skip_ws();

        let mut shape: Option<Vec<u32>> = None;
        let mut offsets: Option<(u64, u64)> = None;

        if self.try_byte(b'}') {
            return Err(Error::broken_data(self.path, 0));
        }

        loop {
            let key = self.parse_string()?;
            self.skip_ws();
            self.expect_byte(b':')?;
            self.skip_ws();

            match key.as_str() {
                "shape" => {
                    shape = Some(self.parse_u32_array()?);
                }
                "data_offsets" => {
                    offsets = Some(self.parse_u64_pair()?);
                }
                _ => {
                    self.skip_json_value()?;
                }
            }

            self.skip_ws();
            if self.try_byte(b',') {
                self.skip_ws();
                continue;
            }

            self.expect_byte(b'}')?;
            break;
        }

        let shape = shape.ok_or_else(|| Error::broken_data(self.path, 0))?;
        let (offset_start, offset_end) = offsets.ok_or_else(|| Error::broken_data(self.path, 0))?;
        if offset_start > offset_end {
            return Err(Error::broken_data(self.path, 0));
        }

        Ok(TensorInfo {
            shape,
            offset: offset_start..offset_end,
        })
    }

    fn parse_u32_array(&mut self) -> Result<Vec<u32>, Error> {
        self.expect_byte(b'[')?;
        self.skip_ws();

        let mut values = Vec::new();
        if self.try_byte(b']') {
            return Ok(values);
        }

        loop {
            values.push(self.parse_u32()?);
            self.skip_ws();

            if self.try_byte(b',') {
                self.skip_ws();
                continue;
            }

            self.expect_byte(b']')?;
            break;
        }

        Ok(values)
    }

    fn parse_u64_pair(&mut self) -> Result<(u64, u64), Error> {
        self.expect_byte(b'[')?;
        self.skip_ws();
        let start = self.parse_u64()?;
        self.skip_ws();
        self.expect_byte(b',')?;
        self.skip_ws();
        let end = self.parse_u64()?;
        self.skip_ws();
        self.expect_byte(b']')?;
        Ok((start, end))
    }

    fn parse_string(&mut self) -> Result<String, Error> {
        self.expect_byte(b'"')?;
        let start = self.idx;

        while self.idx < self.text.len() {
            let b = self.text[self.idx];
            if b == b'\\' {
                self.idx += 2;
                continue;
            }
            if b == b'"' {
                let raw = &self.text[start..self.idx];
                self.idx += 1;
                return String::from_utf8(raw.to_vec())
                    .map_err(|_| Error::broken_data(self.path, 0));
            }
            self.idx += 1;
        }

        Err(Error::broken_data(self.path, 0))
    }

    fn parse_u32(&mut self) -> Result<u32, Error> {
        let value = self.parse_u64()?;
        u32::try_from(value).map_err(|_| Error::broken_data(self.path, 0))
    }

    fn parse_u64(&mut self) -> Result<u64, Error> {
        if self.idx >= self.text.len() || !self.text[self.idx].is_ascii_digit() {
            return Err(Error::broken_data(self.path, 0));
        }

        let mut value = 0u64;
        while self.idx < self.text.len() {
            let b = self.text[self.idx];
            if !b.is_ascii_digit() {
                break;
            }

            value = value
                .checked_mul(10)
                .and_then(|x| x.checked_add((b - b'0') as u64))
                .ok_or_else(|| Error::broken_data(self.path, 0))?;
            self.idx += 1;
        }

        Ok(value)
    }

    fn skip_json_value(&mut self) -> Result<(), Error> {
        self.skip_ws();
        match self.peek_byte() {
            Some(b'{') => self.skip_json_object(),
            Some(b'[') => self.skip_json_array(),
            Some(b'"') => self.skip_json_string(),
            Some(b'-') | Some(b'0'..=b'9') => self.skip_json_number(),
            Some(b't') => self.expect_keyword(b"true"),
            Some(b'f') => self.expect_keyword(b"false"),
            Some(b'n') => self.expect_keyword(b"null"),
            _ => Err(Error::broken_data(self.path, 0)),
        }
    }

    fn skip_json_object(&mut self) -> Result<(), Error> {
        self.expect_byte(b'{')?;
        self.skip_ws();

        if self.try_byte(b'}') {
            return Ok(());
        }

        loop {
            self.skip_json_string()?;
            self.skip_ws();
            self.expect_byte(b':')?;
            self.skip_ws();
            self.skip_json_value()?;
            self.skip_ws();

            if self.try_byte(b',') {
                self.skip_ws();
                continue;
            }

            self.expect_byte(b'}')?;
            break;
        }

        Ok(())
    }

    fn skip_json_array(&mut self) -> Result<(), Error> {
        self.expect_byte(b'[')?;
        self.skip_ws();

        if self.try_byte(b']') {
            return Ok(());
        }

        loop {
            self.skip_json_value()?;
            self.skip_ws();

            if self.try_byte(b',') {
                self.skip_ws();
                continue;
            }

            self.expect_byte(b']')?;
            break;
        }

        Ok(())
    }

    fn skip_json_string(&mut self) -> Result<(), Error> {
        self.expect_byte(b'"')?;
        while self.idx < self.text.len() {
            let b = self.text[self.idx];
            if b == b'\\' {
                self.idx += 2;
                continue;
            }

            self.idx += 1;
            if b == b'"' {
                return Ok(());
            }
        }

        Err(Error::broken_data(self.path, 0))
    }

    fn skip_json_number(&mut self) -> Result<(), Error> {
        if self.try_byte(b'-')
            && (self.idx >= self.text.len() || !self.text[self.idx].is_ascii_digit())
        {
            return Err(Error::broken_data(self.path, 0));
        }

        let mut has_digit = false;
        while self.idx < self.text.len() && self.text[self.idx].is_ascii_digit() {
            has_digit = true;
            self.idx += 1;
        }
        if !has_digit {
            return Err(Error::broken_data(self.path, 0));
        }

        if self.try_byte(b'.') {
            if self.idx >= self.text.len() || !self.text[self.idx].is_ascii_digit() {
                return Err(Error::broken_data(self.path, 0));
            }
            while self.idx < self.text.len() && self.text[self.idx].is_ascii_digit() {
                self.idx += 1;
            }
        }

        if self.try_byte(b'e') || self.try_byte(b'E') {
            self.try_byte(b'+');
            self.try_byte(b'-');
            if self.idx >= self.text.len() || !self.text[self.idx].is_ascii_digit() {
                return Err(Error::broken_data(self.path, 0));
            }
            while self.idx < self.text.len() && self.text[self.idx].is_ascii_digit() {
                self.idx += 1;
            }
        }

        Ok(())
    }

    fn expect_keyword(&mut self, keyword: &[u8]) -> Result<(), Error> {
        let end = self.idx + keyword.len();
        if end > self.text.len() || &self.text[self.idx..end] != keyword {
            return Err(Error::broken_data(self.path, 0));
        }
        self.idx = end;
        Ok(())
    }

    fn skip_ws(&mut self) {
        while self.idx < self.text.len() && self.text[self.idx].is_ascii_whitespace() {
            self.idx += 1;
        }
    }

    fn expect_byte(&mut self, expected: u8) -> Result<(), Error> {
        if self.idx >= self.text.len() || self.text[self.idx] != expected {
            return Err(Error::broken_data(self.path, 0));
        }
        self.idx += 1;
        Ok(())
    }

    fn try_byte(&mut self, expected: u8) -> bool {
        if self.idx < self.text.len() && self.text[self.idx] == expected {
            self.idx += 1;
            return true;
        }
        false
    }

    fn peek_byte(&self) -> Option<u8> {
        self.text.get(self.idx).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const WEIGHT_PATH: &'static str = "model/model.safetensors";

    fn assert(tensor_info: &TensorInfo, expected_shape: &[u32], expected_offset: Range<u64>) {
        assert_eq!(
            tensor_info.shape, expected_shape,
            "actual: {:?}, expected: {:?}",
            tensor_info.shape, expected_shape
        );
        assert_eq!(
            tensor_info.offset, expected_offset,
            "actual: {:?}, expected: {:?}",
            tensor_info.offset, expected_offset
        );
    }

    fn build_weight_info() -> WeightInfo {
        let weight_text =
            WeightText::new(WEIGHT_PATH).expect("initializing weight text should secceed");
        let (weight_info, _) = weight_text.parse().expect("parsing should secceed");

        weight_info
    }

    #[test]
    fn case01_text_embed_tokens_weight() {
        let weight_info = build_weight_info();
        assert(
            &weight_info.embed_tokens_weight,
            &[151936, 1536],
            0..466747392,
        );
    }

    #[test]
    fn case02_text_q_proj_bias() {
        let weight_info = build_weight_info();
        assert(
            &weight_info.layers[0].q_proj_bias,
            &[1536],
            466747392..466750464,
        );
    }

    #[test]
    fn case03_text_k_proj_weight() {
        let weight_info = build_weight_info();
        assert(
            &weight_info.layers[1].k_proj_weight,
            &[256, 1536],
            565065728..565852160,
        );
    }

    #[test]
    fn case04_text_v_proj_weight() {
        let weight_info = build_weight_info();
        assert(
            &weight_info.layers[2].v_proj_weight,
            &[256, 1536],
            659447808..660234240,
        );
    }

    #[test]
    fn case05_text_o_proj_weight() {
        let weight_info = build_weight_info();
        assert(
            &weight_info.layers[3].o_proj_weight,
            &[1536, 1536],
            753829888..758548480,
        );
    }

    #[test]
    fn case06_text_gate_proj_weight() {
        let weight_info = build_weight_info();
        assert(
            &weight_info.layers[4].gate_proj_weight,
            &[8960, 1536],
            852144128..879669248,
        );
    }

    #[test]
    fn case07_text_up_proj_weight() {
        let weight_info = build_weight_info();
        assert(
            &weight_info.layers[5].up_proj_weight,
            &[8960, 1536],
            973264896..1000790016,
        );
    }

    #[test]
    fn case08_text_down_proj_weight() {
        let weight_info = build_weight_info();
        assert(
            &weight_info.layers[6].down_proj_weight,
            &[1536, 8960],
            1094385664..1121910784,
        );
    }

    #[test]
    fn case09_text_input_layernorm_weight() {
        let weight_info = build_weight_info();
        assert(
            &weight_info.layers[7].input_layernorm_weight,
            &[1536],
            1215506432..1215509504,
        );
    }

    #[test]
    fn case10_text_post_attention_layernorm_weight() {
        let weight_info = build_weight_info();
        assert(
            &weight_info.layers[27].post_attention_layernorm_weight,
            &[1536],
            3087422464..3087425536,
        );
    }

    #[test]
    fn case11_text_norm_weight() {
        let weight_info = build_weight_info();
        assert(&weight_info.norm_weight, &[1536], 3087425536..3087428608);
    }

    #[test]
    fn case12_text_lm_head_weight() {
        let weight_info = build_weight_info();
        assert(
            &weight_info.lm_head_weight,
            &[151936, 1536],
            3087428608..3554176000,
        );
    }
}
