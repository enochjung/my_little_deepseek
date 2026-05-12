use super::Format;
use crate::storage::Host;
use std::ops::Range;

pub enum WeightFormat<'a> {
    Safetensor { path: &'a str },
}

pub(crate) struct TensorInfo {
    pub(crate) name: String,
    pub(crate) shape: Vec<u32>,
    pub(crate) offset: Range<usize>,
}

impl Format for WeightFormat<'_> {
    type Output<'a> = TensorInfo;
    type Parser = for<'a> fn(&'a Host) -> Box<dyn Iterator<Item = Self::Output<'a>> + 'a>;

    fn read(&self) -> Result<(Host, Self::Parser), crate::Error> {
        match self {
            &WeightFormat::Safetensor { path } => {
                let file = Host::try_from(path)?;
                Ok((file, safetensor::parse))
            }
        }
    }
}

mod safetensor {
    use super::TensorInfo;
    use crate::storage::Host;

    const HLEN_END_OFFSET: usize = 8;

    pub fn parse(file: &Host) -> Box<dyn Iterator<Item = TensorInfo> + '_> {
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

    fn parse_tensor_info(section: Section, additional_offset: usize) -> TensorInfo {
        let name = str::from_utf8(section.name).unwrap().to_string();
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

        TensorInfo {
            name,
            shape,
            offset,
        }
    }
}

/*
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

    fn build_weight_text() -> WeightText {
        WeightText::new(WEIGHT_PATH).expect("initializing weight text should secceed")
    }

    fn get_weight_info<'a>(weight_text: &'a WeightText) -> &'a WeightInfo {
        let (weight_info, _) = weight_text.parse().expect("parsing should secceed");
        weight_info
    }

    #[test]
    fn case01_text_embed_tokens_weight() {
        let weight_text = build_weight_text();
        let weight_info = get_weight_info(&weight_text);
        assert(
            &weight_info.embed_tokens_weight,
            &[151936, 1536],
            0..466747392,
        );
    }

    #[test]
    fn case02_text_q_proj_bias() {
        let weight_text = build_weight_text();
        let weight_info = get_weight_info(&weight_text);
        assert(
            &weight_info.layers[0].q_proj_bias,
            &[1536],
            466747392..466750464,
        );
    }

    #[test]
    fn case03_text_k_proj_weight() {
        let weight_text = build_weight_text();
        let weight_info = get_weight_info(&weight_text);
        assert(
            &weight_info.layers[1].k_proj_weight,
            &[256, 1536],
            565065728..565852160,
        );
    }

    #[test]
    fn case04_text_v_proj_weight() {
        let weight_text = build_weight_text();
        let weight_info = get_weight_info(&weight_text);
        assert(
            &weight_info.layers[2].v_proj_weight,
            &[256, 1536],
            659447808..660234240,
        );
    }

    #[test]
    fn case05_text_o_proj_weight() {
        let weight_text = build_weight_text();
        let weight_info = get_weight_info(&weight_text);
        assert(
            &weight_info.layers[3].o_proj_weight,
            &[1536, 1536],
            753829888..758548480,
        );
    }

    #[test]
    fn case06_text_gate_proj_weight() {
        let weight_text = build_weight_text();
        let weight_info = get_weight_info(&weight_text);
        assert(
            &weight_info.layers[4].gate_proj_weight,
            &[8960, 1536],
            852144128..879669248,
        );
    }

    #[test]
    fn case07_text_up_proj_weight() {
        let weight_text = build_weight_text();
        let weight_info = get_weight_info(&weight_text);
        assert(
            &weight_info.layers[5].up_proj_weight,
            &[8960, 1536],
            973264896..1000790016,
        );
    }

    #[test]
    fn case08_text_down_proj_weight() {
        let weight_text = build_weight_text();
        let weight_info = get_weight_info(&weight_text);
        assert(
            &weight_info.layers[6].down_proj_weight,
            &[1536, 8960],
            1094385664..1121910784,
        );
    }

    #[test]
    fn case09_text_input_layernorm_weight() {
        let weight_text = build_weight_text();
        let weight_info = get_weight_info(&weight_text);
        assert(
            &weight_info.layers[7].input_layernorm_weight,
            &[1536],
            1215506432..1215509504,
        );
    }

    #[test]
    fn case10_text_post_attention_layernorm_weight() {
        let weight_text = build_weight_text();
        let weight_info = get_weight_info(&weight_text);
        assert(
            &weight_info.layers[27].post_attention_layernorm_weight,
            &[1536],
            3087422464..3087425536,
        );
    }

    #[test]
    fn case11_text_norm_weight() {
        let weight_text = build_weight_text();
        let weight_info = get_weight_info(&weight_text);
        assert(&weight_info.norm_weight, &[1536], 3087425536..3087428608);
    }

    #[test]
    fn case12_text_lm_head_weight() {
        let weight_text = build_weight_text();
        let weight_info = get_weight_info(&weight_text);
        assert(
            &weight_info.lm_head_weight,
            &[151936, 1536],
            3087428608..3554176000,
        );
    }
}

*/
