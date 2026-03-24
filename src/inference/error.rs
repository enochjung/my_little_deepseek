#[derive(Debug)]
pub struct Error {
    kind: ErrorKind,
}

impl Error {
    pub fn io(path: &str, err: std::io::Error) -> Self {
        Self {
            kind: ErrorKind::Io {
                path: path.to_string(),
                err,
            },
        }
    }

    pub fn broken_data(path: &str, line: usize) -> Self {
        Self {
            kind: ErrorKind::BrokenData {
                path: path.to_string(),
                line,
            },
        }
    }

    pub fn unknown_format(path: &str) -> Self {
        Self {
            kind: ErrorKind::UnknownFormat {
                path: path.to_string(),
            },
        }
    }

    pub fn data_not_provided(name: &str) -> Self {
        Self {
            kind: ErrorKind::DataNotProvided {
                name: name.to_string(),
            },
        }
    }

    pub fn invalid_char(codepoint: u32) -> Self {
        Self {
            kind: ErrorKind::InvalidChar { codepoint },
        }
    }
}

impl std::error::Error for Error {}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.kind)
    }
}

#[derive(Debug)]
pub enum ErrorKind {
    Io { path: String, err: std::io::Error },
    BrokenData { path: String, line: usize },
    UnknownFormat { path: String },
    DataNotProvided { name: String },
    InvalidChar { codepoint: u32 },
}

impl std::fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io { path, err } => write!(f, "cannot read {path}: {err}"),
            Self::BrokenData { path, line } => {
                write!(f, "broken {path} at line {line}")
            }
            Self::UnknownFormat { path } => write!(f, "unknown {path} format"),
            Self::DataNotProvided { name } => write!(f, "{name} data not provided"),
            Self::InvalidChar { codepoint } => write!(f, "invalid character: U+{:04X}", codepoint),
        }
    }
}
