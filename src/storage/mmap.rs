use core::ffi::c_void;
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

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn as_ptr(&self) -> *const () {
        self.ptr
    }

    pub(crate) fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr as *const u8, self.len) }
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

impl MmapMut {
    pub(crate) fn new(len: usize) -> Result<Self, crate::Error> {
        let ptr = new_mmap(-1, len, false).map_err(crate::Error::io)?;
        Ok(Self {
            ptr: ptr as *mut (),
            len,
        })
    }

    pub(crate) fn resize(&mut self, len: usize) -> Result<(), crate::Error> {
        let ptr = resize_mmap(self.ptr, self.len, len).map_err(crate::Error::io)?;
        self.ptr = ptr;
        self.len = len;
        Ok(())
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn as_ptr(&self) -> *const () {
        self.ptr
    }

    pub(crate) fn as_mut_ptr(&mut self) -> *mut () {
        self.ptr as *mut ()
    }

    pub(crate) fn as_const_mmap(&self) -> &Mmap {
        unsafe { &*(self as *const MmapMut as *const Mmap) }
    }
    /*
    pub(crate) fn to_readonly(self) -> Mmap {
        let ptr = self.ptr as *const ();
        let len = self.len;
        std::mem::forget(self);
        Mmap { ptr, len }
    }
    */
}

impl From<MmapMut> for Mmap {
    fn from(value: MmapMut) -> Self {
        let ptr = value.ptr;
        let len = value.len;
        std::mem::forget(value);
        Self { ptr, len }
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
    if new_len == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "mmap length must be greater than zero",
        ));
    }
    let flags = libc::MREMAP_MAYMOVE;
    let ptr = unsafe { libc::mremap(ptr as *mut c_void, prev_len, new_len, flags) };
    if ptr == libc::MAP_FAILED {
        return Err(std::io::Error::last_os_error());
    }
    Ok(ptr as *mut ())
}
