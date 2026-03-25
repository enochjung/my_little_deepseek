use super::{Binary, Error, Text};
use crate::inference::utils;
use std::fs::File;
use std::{collections::HashMap, ops::Range};

const NUM_LAYER: usize = 28;

pub struct TensorInfo {
    pub shape: Vec<u32>,
    pub offset: Range<u64>,
}

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

pub struct WeightsInfo {
    pub embed_tokens_weight: TensorInfo,
    pub layers: [LayerInfo; NUM_LAYER],
    pub norm_weight: TensorInfo,
    pub lm_head_weight: TensorInfo,
}

#[allow(unused)]
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

#[allow(unused)]
impl Text for WeightText {
    type Output<'a> = (WeightsInfo, &'a [u16]);

    fn parse(&self) -> Result<Self::Output<'_>, Error> {
        let raw = self.mmap.as_slice();
        let (header_range, payload_range) = split_sections(raw, &self.path)?;

        let header = &raw[header_range];
        let tensors = parse_header(header, &self.path)?;

        let weights_info = WeightsInfo::new(tensors)?;
        let payload_u16 = self
            .mmap
            .get_u16_slice(payload_range)
            .ok_or_else(|| Error::broken_data(&self.path, 0))?;

        Ok((weights_info, payload_u16))
    }
}

// Splits a safetensors blob into byte ranges for JSON header and binary payload.
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

    // Require both header and payload sections to exist.
    if header_end >= raw.len() {
        return Err(Error::broken_data(path, 0));
    }

    let header = &raw[header_start..header_end];
    if header.first() != Some(&b'{') || header.last() != Some(&b'}') {
        return Err(Error::broken_data(path, 0));
    }

    Ok((header_start..header_end, header_end..raw.len()))
}

// Parses safetensors JSON metadata into tensor info map without external crates.
fn parse_header(header: &[u8], path: &str) -> Result<HashMap<String, TensorInfo>, Error> {
    HeaderParser::new(header, path).parse_tensor_map()
}

// Removes a required tensor entry from metadata map.
fn take_tensor(tensors: &mut HashMap<String, TensorInfo>, key: &str) -> Result<TensorInfo, Error> {
    tensors
        .remove(key)
        .ok_or_else(|| Error::data_not_provided(key))
}

impl WeightsInfo {
    // Builds complete `WeightsInfo` (all 28 layers and all fields) from metadata keys.
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
            .map_err(|_| Error::broken_data("weights", 0))?;

        Ok(Self {
            embed_tokens_weight,
            layers,
            norm_weight,
            lm_head_weight,
        })
    }
}

#[allow(unused)]
pub struct WeightBinary {
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
