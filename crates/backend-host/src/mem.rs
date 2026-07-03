use common::{Error, Memory, MemoryMut, MemoryOwn};

use std::os::fd::AsRawFd;

/// A hardware abstraction representing the CPU memory.
///
/// The `HostMem` struct provides the foundational memory management and computational backend
/// for tensor operations running on the host system. It handles allocating and mapping
/// physical storage (e.g., memory-mapped static model weights) and dynamic state buffers
/// (such as the KV Cache).
///
/// In the current architecture, it serves safely as both the Embedding Device (`ED`)
/// and Transformer Device (`TD`).
#[repr(C)]
pub struct HostMem<T> {
    ptr: *mut T,
    size: usize,
}

impl<T> HostMem<T> {
    pub fn as_slice(&self) -> &[T] {
        let len = self.size / size_of::<T>();
        unsafe { std::slice::from_raw_parts(self.ptr, len) }
    }
}

impl<T> TryFrom<std::fs::File> for HostMem<T> {
    type Error = Error;

    fn try_from(file: std::fs::File) -> Result<Self, Self::Error> {
        let metadata = file.metadata().map_err(Error::io)?;
        let fd = file.as_raw_fd();
        let size = (metadata.len() as usize).next_multiple_of(size_of::<T>());
        let ptr = file_mmap(fd, size)? as *mut T;
        Ok(Self { ptr, size })
    }
}

impl<T> Drop for HostMem<T> {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr as *mut libc::c_void, self.size);
        }
    }
}

unsafe impl<T> Send for HostMem<T> {}
unsafe impl<T> Sync for HostMem<T> {}

impl<T> Memory for HostMem<T> {
    type Item = T;
    type Base = Self;

    fn as_base(&self) -> &Self::Base {
        self
    }
    fn size(&self) -> usize {
        self.size
    }
    fn as_ptr(&self) -> *const Self::Item {
        self.ptr
    }
}

impl<T> MemoryMut for HostMem<T> {
    fn as_mut_base(&mut self) -> &mut Self::Base {
        self
    }
    fn as_mut_ptr(&mut self) -> *mut Self::Item {
        self.ptr
    }
}

impl<T> MemoryOwn for HostMem<T> {
    fn new(size: usize) -> Result<Self, Error> {
        let size = size.next_multiple_of(size_of::<T>());
        let ptr = new_mmap(size)? as *mut T;
        Ok(Self { ptr, size })
    }

    fn resize(&mut self, size: usize) -> Result<(), Error> {
        let size = size.next_power_of_two().next_multiple_of(size_of::<T>());
        let ptr = resize_mmap(self.ptr as *mut (), self.size, size)? as *mut T;
        self.ptr = ptr;
        self.size = size;
        Ok(())
    }
}

fn new_mmap(size: usize) -> Result<*mut (), Error> {
    let prot = libc::PROT_READ | libc::PROT_WRITE;
    let flags = libc::MAP_PRIVATE | libc::MAP_ANONYMOUS;
    let ptr = unsafe { libc::mmap(std::ptr::null_mut(), size, prot, flags, -1, 0) };
    if ptr == libc::MAP_FAILED {
        return Err(Error::memory_allocation_failed(size));
    }
    Ok(ptr as *mut ())
}

fn file_mmap(fd: i32, size: usize) -> Result<*mut (), Error> {
    if size == 0 {
        return Err(Error::memory_allocation_failed(0));
    }
    let prot = libc::PROT_READ | libc::PROT_WRITE;
    let flags = libc::MAP_PRIVATE;
    let ptr = unsafe { libc::mmap(std::ptr::null_mut(), size, prot, flags, fd, 0) };
    if ptr == libc::MAP_FAILED {
        return Err(Error::insufficient_storage_space(size, 0));
    }
    Ok(ptr as *mut ())
}

fn resize_mmap(ptr: *mut (), prev_size: usize, new_size: usize) -> Result<*mut (), Error> {
    if new_size == 0 {
        return Err(Error::memory_allocation_failed(0));
    }
    #[cfg(target_os = "linux")]
    {
        let flags = libc::MREMAP_MAYMOVE;
        let ptr = unsafe { libc::mremap(ptr as *mut libc::c_void, prev_size, new_size, flags) };
        if ptr == libc::MAP_FAILED {
            return Err(Error::memory_allocation_failed(new_size));
        }
        Ok(ptr as *mut ())
    }
    #[cfg(not(target_os = "linux"))]
    {
        let new_ptr = new_mmap(new_size)?;
        unsafe {
            let copy_size = prev_size.min(new_size);
            std::ptr::copy_nonoverlapping(ptr as *const u8, new_ptr as *mut u8, copy_size);
            libc::munmap(ptr, prev_size);
        }
        Ok(new_ptr as *mut ())
    }
}
