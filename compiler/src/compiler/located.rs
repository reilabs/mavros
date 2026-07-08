use std::{
    fmt::{self, Debug, Display, Formatter},
    ops::{Deref, DerefMut},
    sync::Arc,
};

use fm::{FileId, FileMap};

/// A value bundled with its source location.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Located<T> {
    payload: T,
    location: SourceLocation,
}

impl<T> Located<T> {
    pub fn new(payload: T, location: SourceLocation) -> Self {
        Self { payload, location }
    }

    pub fn payload(self) -> T {
        self.payload
    }

    pub fn take(self) -> (T, SourceLocation) {
        (self.payload, self.location)
    }

    pub fn location(&self) -> &SourceLocation {
        &self.location
    }

    pub fn location_mut(&mut self) -> &mut SourceLocation {
        &mut self.location
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

impl<T: Display> Display for Located<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.payload.fmt(f)
    }
}

/// Lines and columns, when available, are expected to be 1-based.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
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
    /// a string so synthetic source names can also be represented.
    pub file: Arc<str>,
    pub start: SourcePosition,
    pub end: SourcePosition,
}

impl SourceLocation {
    #[cfg(test)]
    pub fn test() -> Self {
        Self::new(
            "<test>",
            SourcePosition::new(1, 1),
            SourcePosition::new(1, 1),
        )
    }

    /// The location for compiler-synthesized code with no user-source anchor, such as generated
    /// helper functions and dispatch stubs. `origin` names the generator, e.g. a pass or helper
    /// name, and becomes the synthetic source identifier `<origin>`.
    pub fn synthetic(origin: impl AsRef<str>) -> Self {
        Self::new(
            format!("<{}>", origin.as_ref()),
            SourcePosition::new(1, 1),
            SourcePosition::new(1, 1),
        )
    }

    pub fn new(file: impl Into<Arc<str>>, start: SourcePosition, end: SourcePosition) -> Self {
        Self {
            file: file.into(),
            start,
            end,
        }
    }

    pub fn from_file_offsets(
        file: impl Into<Arc<str>>,
        file_map: &FileMap,
        file_id: FileId,
        source: &str,
        start_offset: u32,
        end_offset: u32,
    ) -> Self {
        let start = Self::source_position(file_map, file_id, source, start_offset);
        let end = Self::source_position(file_map, file_id, source, end_offset);

        Self::new(file, start, end)
    }

    /// Resolve a byte offset to a 1-based line/column.
    ///
    /// Lines come from the file map's precomputed line index (`codespan`), so this is a binary
    /// search rather than a scan from the start of the file. The column is the 1-based count of
    /// `char`s from the start of the line to `byte_offset`.
    fn source_position(
        file_map: &FileMap,
        file_id: FileId,
        source: &str,
        byte_offset: u32,
    ) -> SourcePosition {
        use fm::codespan_files::Files;

        let byte_offset = Self::char_boundary_offset(source, byte_offset);

        let Ok(line_index) = file_map.line_index(file_id, byte_offset) else {
            return SourcePosition::new(1, 1);
        };
        let Ok(line_range) = file_map.line_range(file_id, line_index) else {
            return SourcePosition::new(1, 1);
        };

        let column = source[line_range.start..byte_offset].chars().count() + 1;
        SourcePosition::new(line_index as u64 + 1, column as u64)
    }

    fn char_boundary_offset(source: &str, byte_offset: u32) -> usize {
        let mut byte_offset = (byte_offset as usize).min(source.len());
        while byte_offset > 0 && !source.is_char_boundary(byte_offset) {
            byte_offset -= 1;
        }
        byte_offset
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
    use fm::FileManager;
    use std::path::{Path, PathBuf};

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
        let mut located = Located::new(1, location.clone());

        assert_eq!(located.location(), &location);
        assert_eq!(located.location(), &location);
        located.location_mut().start.line = 5;
        location.start.line = 5;
        assert_eq!(AsRef::<i32>::as_ref(&located), &1);

        let located_ref = located.to_ref();
        assert_eq!(*located_ref, &1);
        assert_eq!(located_ref.location(), &location);

        assert_eq!(located.take(), (1, location));
    }

    #[test]
    fn located_can_return_just_payload() {
        let located = Located::new(1, test_location());

        assert_eq!(located.payload(), 1);
    }

    #[test]
    fn source_location_from_file_offsets_uses_codespan_lines_and_char_columns() {
        let mut file_manager = FileManager::new(&PathBuf::new());
        let source = "alpha\nlet x = 1;\n".to_string();
        let file_id = file_manager
            .add_file_with_source(Path::new("main.nr"), source.clone())
            .unwrap();
        let start = source.find("x").unwrap() as u32;
        let end = start + "x = 1".len() as u32;

        let location = SourceLocation::from_file_offsets(
            "main.nr",
            file_manager.as_file_map(),
            file_id,
            &source,
            start,
            end,
        );

        assert_eq!(&*location.file, "main.nr");
        assert_eq!(location.start, SourcePosition::new(2, 5));
        assert_eq!(location.end, SourcePosition::new(2, 10));
    }

    #[test]
    fn source_location_from_file_offsets_clamps_to_char_boundary() {
        let mut file_manager = FileManager::new(&PathBuf::new());
        let source = "éx".to_string();
        let file_id = file_manager
            .add_file_with_source(Path::new("main.nr"), source.clone())
            .unwrap();

        let location = SourceLocation::from_file_offsets(
            "main.nr",
            file_manager.as_file_map(),
            file_id,
            &source,
            1,
            2,
        );

        assert_eq!(location.start, SourcePosition::new(1, 1));
        assert_eq!(location.end, SourcePosition::new(1, 2));
    }
}
