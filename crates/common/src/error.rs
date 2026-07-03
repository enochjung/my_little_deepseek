/// The error type for operations in this library.
///
/// This type encapsulates all possible errors that can occur during model
/// initialization, tensor operations, and session-based inference. It provides
/// structured information to help debug issues related to data integrity,
/// configuration, and resource constraints.
#[derive(Debug)]
pub struct Error {
    kind: ErrorKind,
}

impl Error {
    pub fn broken_data(line: usize) -> Self {
        Self {
            kind: ErrorKind::BrokenData { line },
        }
    }

    pub fn data_not_provided(name: &'static str) -> Self {
        Self {
            kind: ErrorKind::DataNotProvided { name },
        }
    }

    pub fn invalid_char(codepoint: u32) -> Self {
        Self {
            kind: ErrorKind::InvalidChar { codepoint },
        }
    }

    pub fn shape_mismatch(expected: usize, actual: usize) -> Self {
        Self {
            kind: ErrorKind::ShapeMismatch { expected, actual },
        }
    }

    pub fn out_of_bound(index: usize, limit: usize) -> Self {
        Self {
            kind: ErrorKind::OutOfBound { index, limit },
        }
    }

    pub fn configure_failed(field: &'static str) -> Self {
        Self {
            kind: ErrorKind::ConfigureFailed { field },
        }
    }

    pub fn insufficient_storage_space(required: usize, actual: usize) -> Self {
        Self {
            kind: ErrorKind::InsufficientStorageSpace { required, actual },
        }
    }

    pub fn matrix_layout_mismatch(expected_row: bool, actual_row: bool) -> Self {
        Self {
            kind: ErrorKind::MatrixLayoutMismatch {
                expected_row,
                actual_row,
            },
        }
    }

    pub fn memory_allocation_failed(size: usize) -> Self {
        Self {
            kind: ErrorKind::MemoryAllocationFailed { size },
        }
    }

    pub fn raw_os_error(raw_os_error: Option<i32>) -> Self {
        Self {
            kind: ErrorKind::RawOsError { raw_os_error },
        }
    }

}

impl core::error::Error for Error {}
impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.kind)
    }
}

/// A list of possible error categories that can occur in the library.
#[derive(Debug)]
enum ErrorKind {
    /// The input data is corrupted or malformed at the specified line.
    BrokenData { line: usize },
    /// Required data for the specified name is missing.
    DataNotProvided { name: &'static str },
    /// An invalid Unicode character codepoint was encountered.
    InvalidChar { codepoint: u32 },
    /// The shape of the tensor does not match the expected dimensions.
    ShapeMismatch { expected: usize, actual: usize },
    /// An attempt was made to access data outside the valid range.
    OutOfBound { index: usize, limit: usize },
    /// Configuration initialization failed for the specified field.
    ConfigureFailed { field: &'static str },
    /// Insufficient storage space was available to accommodate the required data.
    InsufficientStorageSpace { required: usize, actual: usize },
    /// The matrix layout state does not match the expected layout.
    MatrixLayoutMismatch {
        expected_row: bool,
        actual_row: bool,
    },
    /// Memory allocation failed for the requested size.
    MemoryAllocationFailed { size: usize },
    /// A raw OS error occurred.
    RawOsError { raw_os_error: Option<i32> },
}

impl core::fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::BrokenData { line } => {
                write!(f, "data broken at line {line}")
            }
            Self::DataNotProvided { name } => write!(f, "{name} data not provided"),
            Self::InvalidChar { codepoint } => write!(f, "invalid character: U+{:04X}", codepoint),
            Self::ShapeMismatch { expected, actual } => {
                write!(
                    f,
                    "shape mismatch: expected {expected} bytes, got {actual} bytes"
                )
            }
            Self::OutOfBound { index, limit } => {
                write!(f, "out of bound: index {index}, limit {limit}")
            }
            Self::ConfigureFailed { field } => {
                write!(f, "configuration failed at {field}")
            }
            Self::InsufficientStorageSpace { required, actual } => write!(
                f,
                "insufficient storage space: required {required} bytes, got {actual} bytes"
            ),
            Self::MatrixLayoutMismatch {
                expected_row,
                actual_row,
            } => {
                let expected = if *expected_row {
                    "row-major"
                } else {
                    "column-major"
                };
                let actual = if *actual_row {
                    "row-major"
                } else {
                    "column-major"
                };
                write!(
                    f,
                    "matrix layout mismatch: expected {expected}, got {actual}"
                )
            }
            Self::MemoryAllocationFailed { size } => {
                write!(f, "memory allocation failed: requested {size} bytes")
            }
            Self::RawOsError { raw_os_error } => match raw_os_error {
                Some(raw_os_error) => write!(f, "raw OS error: {raw_os_error}"),
                None => write!(f, "raw OS error"),
            },
        }
    }
}
