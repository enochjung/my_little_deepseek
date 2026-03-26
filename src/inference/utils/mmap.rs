use std::fs::File;
use std::os::fd::AsRawFd;

pub struct Mmap {
    ptr: *const u8,
    len: usize,
}

impl Mmap {
    pub fn new(file: &File) -> Result<Self, std::io::Error> {
        let fd = file.as_raw_fd();
        let len = file.metadata()?.len() as usize;

        if len == 0 {
            return Ok(Self {
                ptr: 0 as *const u8,
                len,
            });
        }

        let ptr = unsafe {
            let ptr = libc::mmap(
                0 as *mut libc::c_void,
                len,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                fd,
                0,
            );
            if ptr == libc::MAP_FAILED {
                return Err(std::io::Error::last_os_error());
            }

            ptr as *const u8
        };

        Ok(Self { ptr, len })
    }

    pub fn as_slice(&self) -> &[u8] {
        if self.len == 0 {
            return &[];
        }

        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl Drop for Mmap {
    fn drop(&mut self) {
        if self.len == 0 {
            return;
        }

        unsafe {
            libc::munmap(self.ptr as *mut libc::c_void, self.len);
        }
    }
}
