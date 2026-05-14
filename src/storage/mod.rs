mod host;

pub(crate) use host::Host;

pub(crate) trait Storage {
    fn name(&self) -> &str;
    fn as_slice(&self) -> &[u8];
    fn as_ptr(&self) -> *const u8;
    fn as_mut_ptr(&mut self) -> Result<*mut u8, crate::Error>;
    fn len(&self) -> usize;
    fn memory_copy(
        &mut self,
        dst_offset: usize,
        src: &Self,
        src_offset: usize,
        len: usize,
    ) -> Result<(), crate::Error>;
}
