use std::alloc::{Layout, alloc, dealloc, handle_alloc_error, realloc};
use std::ptr::NonNull;

#[derive(Debug)]
pub struct AlignedBytes {
    ptr: NonNull<u8>,
    len: usize,
    cap: usize,
}

impl AlignedBytes {
    const ALIGN: usize = 64;

    pub fn new(len: usize, cap: usize) -> Self {
        let cap = cap.max(len).max(Self::ALIGN).next_power_of_two();

        let layout = Layout::from_size_align(cap, Self::ALIGN)
            .expect("aligned allocation layout should be valid");
        let raw = unsafe { alloc(layout) };
        if raw.is_null() {
            handle_alloc_error(layout);
        }
        let ptr = NonNull::new(raw).unwrap();

        Self { ptr, len, cap }
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /*
    pub fn set_len(&mut self, len: usize) {
        debug_assert!(len <= self.cap);
        self.len = len;
    }
    */

    fn reserve(&mut self, additional: usize) {
        let needed = self
            .len
            .checked_add(additional)
            .expect("aligned bytes size overflow");
        if needed <= self.cap {
            return;
        }

        let mut new_cap = self.cap.max(64);
        while new_cap < needed {
            new_cap = new_cap
                .checked_mul(2)
                .expect("aligned bytes capacity overflow");
        }
        self.grow_to(new_cap);
    }

    fn grow_to(&mut self, new_cap: usize) {
        debug_assert!(new_cap >= self.len);

        let new_layout = Layout::from_size_align(new_cap, Self::ALIGN)
            .expect("aligned allocation layout should be valid");

        if self.cap == 0 {
            let raw = unsafe { alloc(new_layout) };
            if raw.is_null() {
                handle_alloc_error(new_layout);
            }
            self.ptr = NonNull::new(raw);
            self.cap = new_cap;
            return;
        }

        let old_layout = Layout::from_size_align(self.cap, Self::ALIGN)
            .expect("aligned allocation layout should be valid");
        let old_ptr = self.ptr.expect("existing allocation").as_ptr();
        let raw = unsafe { realloc(old_ptr, old_layout, new_cap) };
        if raw.is_null() {
            handle_alloc_error(new_layout);
        }

        self.ptr = NonNull::new(raw);
        self.cap = new_cap;
    }

    /*
    pub fn extend_from_slice(&mut self, src: &[u8]) {
        if src.is_empty() {
            return;
        }

        self.reserve(src.len());

        let dst = unsafe {
            self.ptr
                .expect("reserved storage must exist")
                .as_ptr()
                .add(self.len)
        };
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len()) };
        self.len += src.len();
    }
    */
}

impl From<&[u8]> for AlignedBytes {
    fn from(data: &[u8]) -> Self {
        let mut out = Self::new(0, data.len());
        out.extend_from_slice(data);
        out
    }
}

impl Clone for AlignedBytes {
    fn clone(&self) -> Self {
        let len = self.len;
        let cap = self.cap;
        let mut cloned = Self::new(len, cap);

        Self::from_slice(self.as_slice())
    }
}

impl Drop for AlignedBytes {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(self.cap, Self::ALIGN)
            .expect("aligned allocation layout should be valid");
        unsafe { dealloc(self.ptr.as_ptr(), layout) };
    }
}
