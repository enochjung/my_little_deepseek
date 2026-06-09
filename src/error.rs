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
    pub(crate) fn io(err: std::io::Error) -> Self {
        Self {
            kind: ErrorKind::Io { err },
        }
    }

    pub(crate) fn broken_data(line: usize) -> Self {
        Self {
            kind: ErrorKind::BrokenData { line },
        }
    }

    pub(crate) fn data_not_provided(name: &str) -> Self {
        Self {
            kind: ErrorKind::DataNotProvided {
                name: name.to_string(),
            },
        }
    }

    pub(crate) fn invalid_char(codepoint: u32) -> Self {
        Self {
            kind: ErrorKind::InvalidChar { codepoint },
        }
    }

    pub(crate) fn shape_mismatch(expected: usize, actual: usize) -> Self {
        Self {
            kind: ErrorKind::ShapeMismatch { expected, actual },
        }
    }

    pub(crate) fn out_of_bound(index: usize, limit: usize) -> Self {
        Self {
            kind: ErrorKind::OutOfBound { index, limit },
        }
    }

    pub(crate) fn configure_failed(field: &str) -> Self {
        Self {
            kind: ErrorKind::ConfigureFailed {
                field: field.to_string(),
            },
        }
    }

    pub(crate) fn insufficient_storage_space(required: usize, actual: usize) -> Self {
        Self {
            kind: ErrorKind::InsufficientStorageSpace { required, actual },
        }
    }
}

impl std::error::Error for Error {}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.kind)
    }
}

/// A list of possible error categories that can occur in the library.
#[derive(Debug)]
enum ErrorKind {
    /// An error occurred during I/O operations.
    Io { err: std::io::Error },
    /// The input data is corrupted or malformed at the specified line.
    BrokenData { line: usize },
    /// Required data for the specified name is missing.
    DataNotProvided { name: String },
    /// An invalid Unicode character codepoint was encountered.
    InvalidChar { codepoint: u32 },
    /// The shape of the tensor does not match the expected dimensions.
    ShapeMismatch { expected: usize, actual: usize },
    /// An attempt was made to access data outside the valid range.
    OutOfBound { index: usize, limit: usize },
    /// Configuration initialization failed for the specified field.
    ConfigureFailed { field: String },
    /// Insufficient storage space was available to accommodate the required data.
    InsufficientStorageSpace { required: usize, actual: usize },
}

impl std::fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io { err } => write!(f, "io error : {err}"),
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
        }
    }
}
