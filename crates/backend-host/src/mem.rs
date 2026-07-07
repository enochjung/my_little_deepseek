use common::{ElemType, Error, Memory, MemoryMut, MemoryRef};

use crate::Host;

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
pub struct HostMem<T: ElemType> {
    pub mmap_ptr: *mut T,
    pub nrow: u32,
    pub ncol: u32,
}

impl<T: ElemType> HostMem<T> {
    fn size(&self) -> usize {
        (self.nrow as usize) * (self.ncol as usize) * size_of::<T>()
    }
}

impl<T: ElemType> Drop for HostMem<T> {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.mmap_ptr as *mut libc::c_void, self.size());
        }
    }
}

unsafe impl<T: ElemType> Send for HostMem<T> {}
unsafe impl<T: ElemType> Sync for HostMem<T> {}

impl MemoryRef for HostMem<f32> {
    type Item = f32;
    type Base = Host<f32>;

    fn shape(&self) -> (u32, u32) {
        (self.nrow, self.ncol)
    }
}

impl MemoryMut for HostMem<f32> {}

impl Memory for HostMem<f32> {
    fn new(nrow: u32, ncol: u32) -> Result<Self, Error> {
        let size = (nrow as usize) * (ncol as usize) * size_of::<f32>();
        let mmap_ptr = new_mmap(size)? as *mut f32;

        Ok(Self {
            mmap_ptr,
            nrow,
            ncol,
        })
    }

    fn resize(&mut self, nrow: u32, ncol: u32) -> Result<(), Error> {
        let new_size = (nrow as usize) * (ncol as usize) * size_of::<f32>();
        let prev_size = self.size();
        let mmap_ptr = resize_mmap(self.mmap_ptr as *mut (), prev_size, new_size)? as *mut f32;

        self.mmap_ptr = mmap_ptr;
        self.nrow = nrow;
        self.ncol = ncol;

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
