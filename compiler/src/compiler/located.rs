use std::{
    error::Error,
    fmt::{self, Debug, Display, Formatter},
    ops::{Deref, DerefMut},
};

/// Lines and columns are expected to be 1-based when available.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SourcePosition {
    pub line: u32,
    pub column: u32,
}

impl SourcePosition {
    pub fn new(line: u32, column: u32) -> Self {
        Self { line, column }
    }
}

/// The source range associated with a located value.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SourceLocation {
    pub file: String,
    pub start: SourcePosition,
    pub end: SourcePosition,
}

impl SourceLocation {
    pub fn new(file: String, start: SourcePosition, end: SourcePosition) -> Self {
        Self { file, start, end }
    }
}

impl Display for SourceLocation {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}:{}:{}-{}:{}",
            self.file, self.start.line, self.start.column, self.end.line, self.end.column
        )
    }
}

pub type Location = Option<SourceLocation>;

/// A value bundled with its optional source location.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Located<T> {
    payload: T,
    location: Location,
}

impl<T> Located<T> {
    pub fn new(payload: T, location: Location) -> Self {
        Self { payload, location }
    }

    pub fn without_location(payload: T) -> Self {
        Self::new(payload, None)
    }

    pub fn with_location(payload: T, location: SourceLocation) -> Self {
        Self::new(payload, Some(location))
    }

    pub fn take(self) -> (T, Location) {
        (self.payload, self.location)
    }

    pub fn location(&self) -> &Location {
        &self.location
    }

    pub fn location_mut(&mut self) -> &mut Location {
        &mut self.location
    }

    pub fn get_location(&self) -> Option<&SourceLocation> {
        self.location().as_ref()
    }

    pub fn as_ref(&self) -> Located<&T> {
        Located::new(&self.payload, self.location.clone())
    }

    pub fn into_parts(self) -> (T, Location) {
        self.take()
    }
}

impl<T> Deref for Located<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.payload
    }
}

impl<T> DerefMut for Located<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.payload
    }
}

impl<T> AsRef<T> for Located<T> {
    fn as_ref(&self) -> &T {
        &self.payload
    }
}

impl<T> From<T> for Located<T> {
    fn from(value: T) -> Self {
        Located::without_location(value)
    }
}

impl<T: Display> Display for Located<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.payload.fmt(f)
    }
}

impl<T: Error> Error for Located<T> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.payload.source()
    }
}

#[cfg(test)]
mod tests {
    use std::{
        error::Error,
        fmt::{self, Display, Formatter},
    };

    use super::{Located, SourceLocation, SourcePosition};

    #[derive(Debug)]
    struct TestError;

    impl Display for TestError {
        fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
            write!(f, "test error")
        }
    }

    impl Error for TestError {}

    fn test_location() -> SourceLocation {
        SourceLocation::new(
            "test.mav".to_string(),
            SourcePosition::new(3, 5),
            SourcePosition::new(3, 10),
        )
    }

    #[test]
    fn located_exposes_location_ref_and_take() {
        let location = test_location();
        let mut located = Located::with_location(1, location.clone());

        assert_eq!(located.location(), &Some(location.clone()));
        assert_eq!(located.get_location(), Some(&location));
        *located.location_mut() = Some(location.clone());
        assert_eq!(AsRef::<i32>::as_ref(&located), &1);

        let located_ref = located.as_ref();
        assert_eq!(*located_ref, &1);
        assert_eq!(located_ref.get_location(), Some(&location));

        assert_eq!(located.take(), (1, Some(location)));
    }

    #[test]
    fn located_error_forwards_display_and_source() {
        let located = Located::without_location(TestError);
        let err: &dyn Error = &located;

        assert_eq!(err.to_string(), "test error");
        assert!(err.source().is_none());
    }
}
