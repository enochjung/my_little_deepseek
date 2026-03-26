use super::Error;

// const BF16_BYTES: usize = 2;

#[allow(unused)]
pub struct Tensor<'a> {
    data: TensorData<'a>,
    shape: [usize; 2],
}

impl<'a> Tensor<'a> {
    #[allow(unused)]
    pub fn borrowed(data: &'a [u8], shape: [usize; 2]) -> Result<Self, Error> {
        Ok(Self {
            data: TensorData::Borrowed(data),
            shape,
        })
    }

    #[allow(unused)]
    pub fn owned(data: Vec<u8>, shape: [usize; 2]) -> Result<Self, Error> {
        Ok(Self {
            data: TensorData::Owned(data),
            shape,
        })
    }

    #[allow(unused)]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }

    #[allow(unused)]
    pub fn as_bytes(&self) -> &[u8] {
        match &self.data {
            TensorData::Borrowed(data) => data,
            TensorData::Owned(data) => data,
        }
    }

    #[allow(unused)]
    pub fn is_borrowed(&self) -> bool {
        matches!(self.data, TensorData::Borrowed(_))
    }
}

#[allow(unused)]
enum TensorData<'a> {
    Borrowed(&'a [u8]),
    Owned(Vec<u8>),
}
