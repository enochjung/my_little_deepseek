use super::{Host, Mutable, Owned, Storage};
use std::fs::File;
use std::os::fd::AsRawFd;

#[repr(C)]
pub(crate) struct Mmap {
    ptr: *const (),
    len: usize,
}

unsafe impl Send for Mmap {}
unsafe impl Sync for Mmap {}

impl Mmap {
    pub(crate) fn new(path: &str) -> Result<Self, crate::Error> {
        let file = File::open(path).map_err(crate::Error::io)?;
        let metadata = file.metadata().map_err(crate::Error::io)?;
        let fd = file.as_raw_fd();
        let len = metadata.len() as usize;
        let ptr = new_mmap(fd, len, true).map_err(crate::Error::io)?;
        Ok(Self { ptr, len })
    }

    pub(crate) fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr as *const u8, self.len) }
    }
}

impl Storage for Mmap {
    type Loc = Host;

    fn len(&self) -> usize {
        self.len
    }
    fn as_ptr(&self) -> *const () {
        self.ptr
    }
}
impl Owned for Mmap {
    type ReadOnly = Mmap;

    fn into_readonly(self) -> Self::ReadOnly {
        self
    }
}
impl Drop for Mmap {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr as *mut libc::c_void, self.len);
        }
    }
}

#[repr(C)]
pub(crate) struct MmapMut {
    ptr: *mut (),
    len: usize,
}

unsafe impl Send for MmapMut {}

impl MmapMut {
    pub(crate) fn new(len: usize) -> Result<Self, crate::Error> {
        let ptr = new_mmap(-1, len, false).map_err(crate::Error::io)?;
        Ok(Self {
            ptr: ptr as *mut (),
            len,
        })
    }
}

impl Storage for MmapMut {
    type Loc = Host;

    fn len(&self) -> usize {
        self.len
    }
    fn as_ptr(&self) -> *const () {
        self.ptr as *const ()
    }
}
impl Mutable for MmapMut {
    fn as_mut_ptr(&mut self) -> *mut () {
        self.ptr as *mut ()
    }
    fn resize(&mut self, len: usize) -> Result<(), crate::Error> {
        let ptr = resize_mmap(self.ptr, self.len, len).map_err(crate::Error::io)?;
        self.ptr = ptr;
        self.len = len;
        Ok(())
    }
}
impl Owned for MmapMut {
    type ReadOnly = Mmap;

    fn into_readonly(self) -> Self::ReadOnly {
        let ptr = self.ptr;
        let len = self.len;
        std::mem::forget(self);
        Self::ReadOnly { ptr, len }
    }
}
impl Drop for MmapMut {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr as *mut libc::c_void, self.len);
        }
    }
}

fn new_mmap(fd: i32, len: usize, readonly: bool) -> Result<*const (), std::io::Error> {
    if len == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "mmap length must be greater than zero",
        ));
    }

    let prot = match readonly {
        true => libc::PROT_READ,
        false => libc::PROT_READ | libc::PROT_WRITE,
    };
    let flags = match readonly {
        true => libc::MAP_PRIVATE,
        false => libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
    };
    let ptr = unsafe { libc::mmap(std::ptr::null_mut(), len, prot, flags, fd, 0) };
    if ptr == libc::MAP_FAILED {
        return Err(std::io::Error::last_os_error());
    }

    Ok(ptr as *const ())
}

fn resize_mmap(ptr: *mut (), prev_len: usize, new_len: usize) -> Result<*mut (), std::io::Error> {
    #[cfg(target_os = "linux")]
    {
        if new_len == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "mmap length must be greater than zero",
            ));
        }
        let flags = libc::MREMAP_MAYMOVE;
        let ptr = unsafe { libc::mremap(ptr as *mut libc::c_void, prev_len, new_len, flags) };
        if ptr == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error());
        }
        Ok(ptr as *mut ())
    }

    #[cfg(not(target_os = "linux"))]
    {
        if new_len == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "mmap length must be greater than zero",
            ));
        }

        let new_ptr_const = new_mmap(-1, new_len, false)?;
        let new_ptr = new_ptr_const as *mut u8;

        unsafe {
            let copy_len = std::cmp::min(prev_len, new_len);
            std::ptr::copy_nonoverlapping(ptr as *const u8, new_ptr as *mut u8, copy_len);
            libc::munmap(ptr as *mut libc::c_void, prev_len);
        }

        Ok(new_ptr as *mut ())
    }
}

#[cfg(test)]
impl From<&[f32]> for Mmap {
    /// Test Helper Function
    fn from(value: &[f32]) -> Self {
        MmapMut::from(value).into_readonly()
    }
}

#[cfg(test)]
impl From<&[f32]> for MmapMut {
    /// Test Helper Function
    fn from(value: &[f32]) -> Self {
        use crate::tensor::{ElemType, F32};

        let len = value.len() * F32::BYTES;
        let mut mmap = MmapMut::new(len).expect("allocating mmap should succeed");

        let src = value.as_ptr();
        let dst = mmap.as_mut_ptr() as *mut f32;
        let count = value.len();
        unsafe { std::ptr::copy_nonoverlapping(src, dst, count) };

        mmap
    }
}
