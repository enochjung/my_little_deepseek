use std::fmt::{Display, Formatter};

#[derive(Debug)]
pub enum AppError {
    Io(std::io::Error),
    InvalidState(&'static str),
    InvalidUnicodeDataLine {
        line: usize,
        reason: &'static str,
    },
    InvalidCompositionExclusionLine {
        line: usize,
        reason: &'static str,
    },
    InvalidHexCodepoint {
        line: usize,
        value: String,
    },
    InvalidCombiningClass {
        line: usize,
        value: String,
    },
}

impl Display for AppError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(err) => write!(f, "I/O error: {err}"),
            Self::InvalidState(msg) => write!(f, "Invalid state: {msg}"),
            Self::InvalidUnicodeDataLine { line, reason } => {
                write!(f, "Invalid UnicodeData.txt line {line}: {reason}")
            }
            Self::InvalidCompositionExclusionLine { line, reason } => {
                write!(f, "Invalid CompositionExclusions.txt line {line}: {reason}")
            }
            Self::InvalidHexCodepoint { line, value } => {
                write!(f, "Invalid hex codepoint at line {line}: {value}")
            }
            Self::InvalidCombiningClass { line, value } => {
                write!(f, "Invalid combining class at line {line}: {value}")
            }
        }
    }
}

impl std::error::Error for AppError {}

impl From<std::io::Error> for AppError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}
