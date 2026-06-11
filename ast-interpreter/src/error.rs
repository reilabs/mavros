use noirc_errors::Location;

/// A failure encountered while interpreting the monomorphized AST.
///
/// `AssertionFailed` is the oracle's signal: the program's own `assert`s encode its expected
/// results, so a failed assertion means the AST does not compute what the source claims.
/// `Unsupported` marks an AST construct the interpreter does not yet evaluate — kept explicit
/// (never silently skipped) so coverage gaps are loud.
#[derive(Debug)]
pub enum InterpretError {
    /// An `assert`/`constrain` evaluated to false.
    AssertionFailed { location: Location, message: Option<String> },
    /// Checked integer arithmetic overflowed the operand type (Noir's `+`/`-`/`*` are checked).
    Overflow(String),
    /// Integer or field division by zero.
    DivisionByZero,
    /// A type mismatch that a well-formed monomorphized AST should never contain.
    Type(String),
    /// An AST construct the interpreter does not yet handle (e.g. closures, intrinsics).
    Unsupported(String),
    /// An invariant the interpreter relies on was violated.
    Internal(String),
}

impl std::fmt::Display for InterpretError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InterpretError::AssertionFailed { message, .. } => match message {
                Some(m) => write!(f, "assertion failed: {m}"),
                None => write!(f, "assertion failed"),
            },
            InterpretError::Overflow(op) => write!(f, "integer overflow in {op}"),
            InterpretError::DivisionByZero => write!(f, "division by zero"),
            InterpretError::Type(m) => write!(f, "type error: {m}"),
            InterpretError::Unsupported(m) => write!(f, "unsupported construct: {m}"),
            InterpretError::Internal(m) => write!(f, "internal interpreter error: {m}"),
        }
    }
}

impl std::error::Error for InterpretError {}
