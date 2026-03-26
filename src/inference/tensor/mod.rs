use super::Error;

#[allow(unused)]
pub struct Tensor<'a> {
    data: TensorData<'a>,
    shape: [usize; 2],
}

impl<'a> Tensor<'a> {
    #[allow(unused)]
    pub fn borrowed(data: &'a [u8], shape: [usize; 2]) -> Result<Self, Error> {
        todo!()
    }

    #[allow(unused)]
    pub fn owned(data: Vec<u8>, shape: [usize; 2]) -> Result<Self, Error> {
        todo!()
    }
}

#[allow(unused)]
enum TensorData<'a> {
    Borrowed(&'a [u8]),
    Owned(Vec<u8>),
}
