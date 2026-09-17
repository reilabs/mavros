//! Basic support for located compiler diagnostics.
//!
//! A [`Diagnostic`] is what is produced when the compiler meets a program that it cannot represent
//! soundly. While most such programs should be refused by the frontend, this infrastructure ensures
//! that we can report user-originated diagnostics.

use std::{
    fmt::{self, Display, Formatter},
    ops::Range,
    sync::Arc,
};

use codespan_reporting::{
    diagnostic::{Diagnostic as CodespanDiagnostic, Label},
    files::SimpleFile,
    term::{self, Chars, Config},
};

use crate::{
    collections::HashMap,
    compiler::located::{SourceLocation, SourcePosition},
};

/// A reason behind a program refusal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Diagnostic {
    /// A one-line statement of the error.
    message: String,

    /// The location in the program to tag the diagnostic with.
    location: SourceLocation,

    /// The text to be written at the span.
    label: Option<String>,

    /// Info needed to act on the error.
    notes: Vec<String>,

    /// The text of the file `location` names, where available.
    source: Option<Arc<str>>,
}

impl Diagnostic {
    /// A diagnostic stating `message`, spanning `location`.
    #[must_use]
    pub fn error(message: impl Into<String>, location: SourceLocation) -> Self {
        Self {
            message: message.into(),
            location,
            label: None,
            notes: Vec::new(),
            source: None,
        }
    }

    /// Render against `source` rather than against whatever the location's path names on disk.
    #[must_use]
    pub fn with_source(mut self, source: Arc<str>) -> Self {
        self.source = Some(source);
        self
    }

    /// Write `label` beside the carets.
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Add a `= note: …` line below the snippet.
    #[must_use]
    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }

    /// Where the diagnostic points.
    pub fn location(&self) -> &SourceLocation {
        &self.location
    }

    /// What the diagnostic says, without its location or snippet.
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Render against [`Self::source`] where the producer supplied it and against `sources`
    /// otherwise, falling back to [`Self::header`] wherever the snippet cannot be drawn.
    fn render(&self, sources: &mut SourceCache) -> String {
        let source = match &self.source {
            Some(source) => Some(source.clone()),
            None => sources.source(&self.location.file),
        };
        let Some(source) = source else {
            return self.header();
        };
        let Some(range) = self.byte_range(&source) else {
            return self.header();
        };

        let file = SimpleFile::new(self.location.file.as_ref(), source.as_ref());
        let mut label = Label::primary((), range);
        if let Some(text) = &self.label {
            label = label.with_message(text);
        }
        let diagnostic = CodespanDiagnostic::error()
            .with_message(&self.message)
            .with_labels(vec![label])
            .with_notes(
                self.notes
                    .iter()
                    .map(|note| format!("note: {note}"))
                    .collect(),
            );

        term::emit_into_string(&rendering_config(), &file, &diagnostic)
            .unwrap_or_else(|_| self.header())
    }

    /// The diagnostic without its snippet: everything that does not need the source to state.
    fn header(&self) -> String {
        let mut text = format!("error: {}\n  --> {}\n", self.message, self.location);
        for note in &self.notes {
            text.push_str(&format!("  = note: {note}\n"));
        }
        text
    }

    /// The byte range the carets cover, or `None` if the source does not have those coordinates.
    fn byte_range(&self, source: &str) -> Option<Range<usize>> {
        let start = byte_offset(source, self.location.start)?;
        let end = byte_offset(source, self.location.end)
            .unwrap_or(start)
            .max(start);
        Some(start..end)
    }
}

impl Display for Diagnostic {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", render_all(std::slice::from_ref(self)))
    }
}

/// Render `diagnostics` in order, sharing one read of each source file.
pub fn render_all(diagnostics: &[Diagnostic]) -> String {
    let mut sources = SourceCache::default();
    diagnostics
        .iter()
        .map(|diagnostic| diagnostic.render(&mut sources))
        .collect::<Vec<_>>()
        .join("\n")
}

/// The sources a set of diagnostics names but does not carry, each read **at most once**.
#[derive(Default)]
struct SourceCache {
    files: HashMap<Arc<str>, Option<Arc<str>>>,
}

impl SourceCache {
    fn source(&mut self, path: &Arc<str>) -> Option<Arc<str>> {
        if let Some(source) = self.files.get(path) {
            return source.clone();
        }

        let source = std::fs::read_to_string(path.as_ref())
            .ok()
            .map(Arc::<str>::from);
        self.files.insert(path.clone(), source.clone());
        source
    }
}

/// ASCII drawing, matching `rustc`'s `-->`, `|` and `^`, to work better with captured outputs.
fn rendering_config() -> Config {
    Config {
        chars: Chars::ascii(),
        ..Config::default()
    }
}

/// The byte offset of a 1-based line/column pair, or `None` if the source has no such line.
fn byte_offset(source: &str, position: SourcePosition) -> Option<usize> {
    let line_index = usize::try_from(position.line).ok()?.checked_sub(1)?;
    let column_index = usize::try_from(position.column).ok()?.saturating_sub(1);

    let mut offset = 0;
    for (index, line) in source.split_inclusive('\n').enumerate() {
        if index == line_index {
            let content = line.trim_end_matches('\n').trim_end_matches('\r');
            let within = content
                .char_indices()
                .nth(column_index)
                .map_or(content.len(), |(byte, _)| byte);
            return Some(offset + within);
        }
        offset += line.len();
    }

    None
}

// TESTS
// ================================================================================================

#[cfg(test)]
mod tests {
    use super::*;

    use std::io::Write as _;

    /// A file holding `source`, and the location of the first occurrence of `span` within it.
    fn located(source: &str, span: &str) -> (tempfile::NamedTempFile, SourceLocation) {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(source.as_bytes()).unwrap();
        file.flush().unwrap();

        let offset = source.find(span).expect("the span is in the source");
        let preceding = &source[..offset];
        let line = preceding.matches('\n').count() + 1;
        let line_start = preceding.rfind('\n').map_or(0, |index| index + 1);
        let column = source[line_start..offset].chars().count() + 1;
        let end_column = column + span.chars().count();

        let location = SourceLocation::new(
            file.path().to_string_lossy().as_ref(),
            SourcePosition::new(line as u64, column as u64),
            SourcePosition::new(line as u64, end_column as u64),
        );

        (file, location)
    }

    #[test]
    fn a_diagnostic_draws_carets_under_its_span() {
        let source = "fn main() {\n    let f = wide as Field;\n}\n";
        let (file, location) = located(source, "wide as Field");
        let path = file.path().to_string_lossy().to_string();

        let rendered = Diagnostic::error("an int254 cannot be cast to Field", location)
            .with_label("2^254 exceeds the bn254 modulus")
            .with_note("the widest integer this field carries injectively is int253")
            .to_string();

        assert_eq!(
            rendered,
            format!(
                "error: an int254 cannot be cast to Field\n  \
                 --> {path}:2:13\n  \
                 |\n\
                 2 |     let f = wide as Field;\n  \
                 |             ^^^^^^^^^^^^^ 2^254 exceeds the bn254 modulus\n  \
                 |\n  \
                 = note: the widest integer this field carries injectively is int253\n\n"
            )
        );
    }

    /// A generated helper has no file to quote, and a diagnostic about one must still say
    /// everything it can rather than failing to render.
    #[test]
    fn a_location_with_no_file_renders_the_header_alone() {
        let rendered = Diagnostic::error(
            "wide multiplication is not supported",
            SourceLocation::synthetic("my_pass"),
        )
        .with_label("this operand is int1000")
        .with_note("support arrives with the wide arithmetic lowering")
        .to_string();

        assert_eq!(
            rendered,
            "error: wide multiplication is not supported\n  \
             --> <my_pass>:1:1\n  \
             = note: support arrives with the wide arithmetic lowering\n"
        );
    }

    /// The column is a count of `char`s, so a line whose prefix is not ASCII still puts the carets
    /// under the span rather than under a byte offset into it.
    #[test]
    fn a_multi_byte_prefix_does_not_shift_the_carets() {
        let source = "let \u{e9}\u{e9}\u{e9} = wide;\n";
        let (_file, location) = located(source, "wide");

        let rendered = Diagnostic::error("nope", location).to_string();
        let caret_line = rendered.lines().nth(4).unwrap();
        let source_line = rendered.lines().nth(3).unwrap();

        // Counted in `char`s: the two lines agree on the column the carets sit at, and the source
        // line's multi-byte characters mean they do not agree on the byte offset of it.
        let carets = caret_line[..caret_line.find('^').unwrap()].chars().count();
        let span = source_line[..source_line.find("wide").unwrap()]
            .chars()
            .count();
        assert_eq!(carets, span, "the carets sit under the span:\n{rendered}");
    }

    /// A span the source cannot have — past the last line — is not a reason to fail to report.
    #[test]
    fn coordinates_the_file_does_not_have_fall_back_to_the_header() {
        let (file, mut location) = located("one line\n", "one");
        location.start.line = 99;
        location.end.line = 99;
        let path = file.path().to_string_lossy().to_string();

        assert_eq!(
            Diagnostic::error("nope", location).to_string(),
            format!("error: nope\n  --> {path}:99:1\n")
        );
    }

    /// The frontend gives a zero-width span for a location it knows only as a point, and the
    /// renderer draws a caret at one rather than nothing. That is the renderer's own behaviour and
    /// not something this module arranges, which is why it is pinned here: nothing else would
    /// notice a version of it that drew an empty run.
    #[test]
    fn a_zero_width_span_still_draws_one_caret() {
        let (_file, mut location) = located("let x = 1;\n", "x");
        location.end = location.start;

        let rendered = Diagnostic::error("nope", location).to_string();
        assert!(rendered.contains("|     ^\n"), "{rendered}");
    }

    /// The end of the last line, with nothing after it to draw a caret over.
    #[test]
    fn a_zero_width_span_at_the_end_of_the_source_renders() {
        let source = "let x = 1;";
        let (_file, mut location) = located(source, "1;");
        location.start = SourcePosition::new(1, source.chars().count() as u64 + 1);
        location.end = location.start;

        let rendered = Diagnostic::error("nope", location).to_string();
        assert!(rendered.contains("^"), "{rendered}");
    }

    /// A carried source is the text the coordinates were measured against, so it wins over
    /// whatever the path names now.
    #[test]
    fn a_carried_source_is_quoted_rather_than_the_file_on_disk() {
        let (_file, location) = located("let stale = 1;\n", "stale");
        let fresh: Arc<str> = Arc::from("let fresh = 1;\n");

        let rendered = Diagnostic::error("nope", location)
            .with_source(fresh)
            .to_string();

        assert!(rendered.contains("let fresh = 1;"), "{rendered}");
        assert!(!rendered.contains("stale"), "{rendered}");
    }

    /// A location no file backs still renders its snippet when the source came with it.
    #[test]
    fn a_carried_source_renders_for_a_path_that_does_not_exist() {
        let location = SourceLocation::new(
            "std/lib.nr",
            SourcePosition::new(1, 5),
            SourcePosition::new(1, 10),
        );

        let rendered = Diagnostic::error("nope", location)
            .with_source(Arc::from("let embedded = 1;\n"))
            .to_string();

        assert!(rendered.contains("let embedded = 1;"), "{rendered}");
        assert!(rendered.contains('^'), "{rendered}");
    }

    #[test]
    fn every_diagnostic_is_rendered() {
        let (_file, location) = located("let x = 1;\n", "x");
        let rendered = render_all(&[
            Diagnostic::error("first", location.clone()),
            Diagnostic::error("second", location),
        ]);

        assert!(rendered.contains("error: first"), "{rendered}");
        assert!(rendered.contains("error: second"), "{rendered}");
    }

    /// A column past the end of a line lands on the line's last character, not in its terminator.
    #[test]
    fn a_column_past_the_line_clamps_to_its_content() {
        let source = "ab\ncd\n";
        assert_eq!(byte_offset(source, SourcePosition::new(1, 99)), Some(2));
        assert_eq!(byte_offset(source, SourcePosition::new(2, 1)), Some(3));
        assert_eq!(byte_offset(source, SourcePosition::new(3, 1)), None);
    }
}
