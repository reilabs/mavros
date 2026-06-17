use std::{
    fmt::{self, Debug, Display, Formatter},
    ops::{Deref, DerefMut},
};

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

    pub fn without(payload: T) -> Self {
        Self::new(payload, None)
    }

    pub fn with(payload: T, location: SourceLocation) -> Self {
        Self::new(payload, Some(location))
    }

    pub fn payload(self) -> T {
        self.payload
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

    pub fn get_location_mut(&mut self) -> Option<&mut SourceLocation> {
        self.location_mut().as_mut()
    }

    pub fn to_ref(&self) -> Located<&T> {
        Located::new(&self.payload, self.location.clone())
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
    // TODO: Remove this once source locations become non-optional.
    fn from(value: T) -> Self {
        Located::without(value)
    }
}

impl<T: Display> Display for Located<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.payload.fmt(f)
    }
}

pub type Location = Option<SourceLocation>;

/// Lines and columns, when available, are expected to be 1-based.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SourcePosition {
    pub line: u64,
    pub column: u64,
}

impl SourcePosition {
    pub fn new(line: u64, column: u64) -> Self {
        Self { line, column }
    }
}

/// The source range associated with a located value.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SourceLocation {
    /// The user-facing source identifier.
    ///
    /// This should be an absolute filesystem path when the source is backed by a file, but remains
    /// a string so compiler-generated and other synthetic sources can also be represented.
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
        write!(f, "{}:{}:{}", self.file, self.start.line, self.start.column)
    }
}

#[cfg(test)]
mod tests {
    use super::{Located, SourceLocation, SourcePosition};

    fn test_location() -> SourceLocation {
        SourceLocation::new(
            "test.mav".to_string(),
            SourcePosition::new(3, 5),
            SourcePosition::new(3, 10),
        )
    }

    #[test]
    fn located_exposes_location_ref_and_take() {
        let mut location = test_location();
        let mut located = Located::with(1, location.clone());

        assert_eq!(located.location(), &Some(location.clone()));
        assert_eq!(located.get_location(), Some(&location));
        located.get_location_mut().unwrap().start.line = 5;
        location.start.line = 5;
        assert_eq!(AsRef::<i32>::as_ref(&located), &1);

        let located_ref = located.to_ref();
        assert_eq!(*located_ref, &1);
        assert_eq!(located_ref.get_location(), Some(&location));

        assert_eq!(located.take(), (1, Some(location)));
    }

    #[test]
    fn located_can_return_just_payload() {
        let located = Located::with(1, test_location());

        assert_eq!(located.payload(), 1);
    }
}
