use std::fs::File;
use std::ops::Range;
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

    pub fn get_u16_slice(&self, range: Range<usize>) -> Option<&[u16]> {
        if range.start > range.end || range.end > self.len {
            return None;
        }

        let byte_len = range.end - range.start;
        if byte_len % 2 != 0 {
            return None;
        }

        let ptr = self.ptr.wrapping_add(range.start);
        if (ptr as usize) % std::mem::align_of::<u16>() != 0 {
            return None;
        }

        Some(unsafe { std::slice::from_raw_parts(ptr as *const u16, byte_len / 2) })
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
