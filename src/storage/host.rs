use super::Storage;
use std::fs::File;

pub(crate) struct Host {
    name: String,
    data: mmap::Mmap,
}

impl Host {
    pub(crate) fn name(&self) -> &str {
        &self.name
    }

    pub(crate) fn as_slice(&self) -> &[u8] {
        self.data.as_slice()
    }
}

impl TryFrom<&str> for Host {
    type Error = crate::Error;

    fn try_from(path: &str) -> Result<Self, Self::Error> {
        let file = File::open(path).map_err(|err| crate::Error::io(path, err))?;
        let data = mmap::Mmap::try_from(file).map_err(|err| crate::Error::io(path, err))?;

        Ok(Self {
            name: path.to_string(),
            data,
        })
    }
}

impl Storage for Host {}

mod mmap {
    use std::fs::File;
    use std::os::fd::AsRawFd;

    pub struct Mmap {
        mtype: MmapType,
        ptr: *mut u8,
        len: usize,
    }

    enum MmapType {
        ReadOnlyFile(File),
        #[allow(unused)]
        WritableFile(File),
        WritableMemory,
    }

    impl Mmap {
        pub fn new(len: usize) -> Result<Self, std::io::Error> {
            if len == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
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
                return Err(std::io::Error::last_os_error());
            }

            let mtype = MmapType::WritableMemory;
            let ptr = ptr as *mut u8;

            Ok(Self { mtype, ptr, len })
        }

        pub fn resize(&mut self, len: usize) -> Result<(), std::io::Error> {
            match self.mtype {
                MmapType::ReadOnlyFile(_) => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::PermissionDenied,
                        "file is read-only",
                    ));
                }
                _ => {}
            }

            if len == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "mmap length must be greater than zero",
                ));
            }

            if len == self.len {
                return Ok(());
            }

            let ptr = unsafe {
                libc::mremap(
                    self.ptr as *mut libc::c_void,
                    self.len,
                    len,
                    libc::MREMAP_MAYMOVE,
                )
            };

            if ptr == libc::MAP_FAILED {
                return Err(std::io::Error::last_os_error());
            }

            self.ptr = ptr as *mut u8;
            self.len = len;

            Ok(())
        }

        pub fn len(&self) -> usize {
            self.len
        }

        pub fn as_ptr(&self) -> *const u8 {
            self.ptr as *const u8
        }

        pub fn as_mut_ptr(&mut self) -> Result<*mut u8, std::io::Error> {
            match self.mtype {
                MmapType::ReadOnlyFile(_) => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::PermissionDenied,
                        "file is read-only",
                    ));
                }
                _ => {}
            }

            Ok(self.ptr)
        }

        pub fn as_slice(&self) -> &[u8] {
            unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
        }
    }

    impl Drop for Mmap {
        fn drop(&mut self) {
            unsafe {
                libc::munmap(self.ptr as *mut libc::c_void, self.len);
            }
        }
    }

    impl TryFrom<File> for Mmap {
        type Error = std::io::Error;

        fn try_from(file: File) -> Result<Self, Self::Error> {
            let metadata = file.metadata()?;
            let readonly = metadata.permissions().readonly();
            let len = metadata.len() as usize;

            if len == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "mmap length must be greater than zero",
                ));
            }

            let fd = file.as_raw_fd();

            let (mtype, ptr) = match readonly {
                true => {
                    let mtype = MmapType::ReadOnlyFile(file);
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

                    (mtype, ptr)
                }
                false => {
                    let mtype = MmapType::WritableFile(file);
                    let ptr = unsafe {
                        libc::mmap(
                            std::ptr::null_mut(),
                            len,
                            libc::PROT_READ | libc::PROT_WRITE,
                            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                            fd,
                            0,
                        )
                    };

                    (mtype, ptr)
                }
            };

            if ptr == libc::MAP_FAILED {
                return Err(std::io::Error::last_os_error());
            }

            let ptr = ptr as *mut u8;

            Ok(Self { mtype, ptr, len })
        }
    }
}
