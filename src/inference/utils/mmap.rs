use std::fs::File;
use std::io;
use std::os::fd::AsRawFd;
use std::ptr::NonNull;

/// Owns a memory-mapped region and exposes its raw pointer.
pub struct Mmap {
    ptr: NonNull<u8>,
    len: usize,
    read_only: bool,
}

impl Mmap {
    /// Allocates a writable anonymous mapping with the requested length.
    ///
    /// Returns an error when `len == 0` or when the OS mapping call fails.
    pub fn new(len: usize) -> Result<Self, io::Error> {
        if len == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "mmap length must be greater than zero",
            ));
        }

        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };

        if ptr == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }

        let ptr = unsafe { NonNull::new_unchecked(ptr as *mut u8) };

        Ok(Self {
            ptr,
            len,
            read_only: false,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    /// Reallocates the mapping to a new length using `mremap`.
    ///
    /// Returns an error when `len == 0` or when resizing fails.
    pub fn resize(&mut self, len: usize) -> Result<(), io::Error> {
        if len == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "mmap length must be greater than zero",
            ));
        }

        if len == self.len {
            return Ok(());
        }

        let ptr = unsafe {
            libc::mremap(
                self.ptr.as_ptr() as *mut libc::c_void,
                self.len,
                len,
                libc::MREMAP_MAYMOVE,
            )
        };

        if ptr == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }

        self.ptr = unsafe { NonNull::new_unchecked(ptr as *mut u8) };
        self.len = len;
        Ok(())
    }

    /// Returns the start address of the mapped region.
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    /// Returns the start address of the mapped region as a mutable pointer.
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        assert!(self.read_only == false, "the memory region is read-only");
        self.ptr.as_ptr()
    }

    /// Returns a slice view of the mapped region.
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Returns a mutable slice view of the mapped region.
    ///
    /// Panics if the mapped region is read-only.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        assert!(self.read_only == false, "the memory region is read-only");
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

/// Creates a file-backed private mapping from the given file.
///
/// Panics when the file metadata cannot be read or the mapping call fails.
impl From<&File> for Mmap {
    fn from(file: &File) -> Self {
        let len = file
            .metadata()
            .expect("failed to read file metadata for mmap")
            .len() as usize;

        assert!(len != 0, "mmap length must be greater than zero");

        let fd = file.as_raw_fd();

        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                fd,
                0,
            )
        };

        assert!(ptr != libc::MAP_FAILED, "failed to map file into memory");

        let ptr = unsafe { NonNull::new_unchecked(ptr as *mut u8) };

        Self {
            ptr,
            len,
            read_only: true,
        }
    }
}

impl Drop for Mmap {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr.as_ptr() as *mut libc::c_void, self.len);
        }
    }
}
