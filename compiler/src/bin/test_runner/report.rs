//! STATUS tables: rendering, parsing and merging them, and the regression and growth checks.

use super::*;

pub(crate) struct ParsedRow {
    pub(crate) name: String,
    pub(crate) field: String,
    pub(crate) cells: Vec<String>,
    pub(crate) rows: Option<usize>,
    pub(crate) cols: Option<usize>,
    pub(crate) size_bytes: Option<usize>,
    pub(crate) vm_debug_size_bytes: Option<usize>,
    pub(crate) wasm_size_bytes: Option<usize>,
    pub(crate) wasm_debug_size_bytes: Option<usize>,
}

pub(crate) const STATUS_COLUMNS: &[&str] = &[
    "Test",
    "Compiled",
    "R1CS",
    "Rows",
    "Cols",
    "Bytecode Size",
    "VM Debug Sidecar Size",
    "WASM Size",
    "WASM Debug Sidecar Size",
    "Compile",
    "Witgen Run VM",
    "Witgen Correct",
    "Witgen No Leak",
    "AD Run VM",
    "AD Correct",
    "AD No Leak",
    "WASM Compile",
    "Witgen WASM Run",
    "Witgen WASM Correct",
    "Witgen WASM No Leak",
    "AD WASM Run",
    "AD WASM Correct",
    "AD WASM No Leak",
    "Field",
];

pub(crate) fn markdown_cells(line: &str) -> Vec<String> {
    line.split('|')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// A parsed STATUS.md: the rows plus the header's column-name → cell-index map. Columns are
/// looked up by name so a baseline and a current file with different layouts (e.g. a baseline
/// predating a column addition) can still be compared column-for-column.
pub(crate) struct StatusTable {
    pub(crate) columns: HashMap<String, usize>,
    pub(crate) rows: Vec<ParsedRow>,
}

impl StatusTable {
    /// The named cell of `row`, or `None` if this table's layout lacks that column.
    pub(crate) fn cell<'a>(&self, row: &'a ParsedRow, name: &str) -> Option<&'a str> {
        self.columns
            .get(name)
            .and_then(|&idx| row.cells.get(idx))
            .map(String::as_str)
    }
}

pub(crate) fn parse_status_table(path: &Path) -> StatusTable {
    let content =
        fs::read_to_string(path).unwrap_or_else(|_| ice_usr!("Cannot read {}", path.display()));
    parse_status_content(&content)
}

pub(crate) fn parse_status_content(content: &str) -> StatusTable {
    let columns: HashMap<String, usize> =
        markdown_cells(content.lines().next().unwrap_or_default())
            .into_iter()
            .enumerate()
            .map(|(idx, name)| (name, idx))
            .collect();

    let mut rows = Vec::new();
    for line in content.lines().skip(2) {
        let cells = markdown_cells(line);
        if cells.len() != columns.len() {
            continue;
        }
        let get = |name: &str| columns.get(name).map(|&idx| cells[idx].as_str());
        let parse_count = |name: &str| get(name).and_then(|cell| cell.parse::<usize>().ok());
        rows.push(ParsedRow {
            name: get("Test").unwrap_or_default().to_string(),
            // Reports before field selection contain BN254 results only.
            field: get("Field").unwrap_or("bn254").to_string(),
            rows: parse_count("Rows"),
            cols: parse_count("Cols"),
            size_bytes: parse_count("Bytecode Size"),
            vm_debug_size_bytes: parse_count("VM Debug Sidecar Size"),
            wasm_size_bytes: parse_count("WASM Size"),
            wasm_debug_size_bytes: parse_count("WASM Debug Sidecar Size"),
            cells,
        });
    }
    StatusTable { columns, rows }
}

/// Merge one complete report per supported field. Refuse missing fields, duplicate rows and
/// different test inventories so CI never publishes a partial corpus as a complete result.
pub(crate) fn merge_status_reports(reports: &[String]) -> Result<String, String> {
    if reports.len() != FieldId::ALL.len() {
        return Err("expected one report per supported field".into());
    }
    let mut seen_fields = Vec::new();
    let mut order = HashMap::default();
    let mut results = Vec::new();
    for (report_index, report) in reports.iter().enumerate() {
        let table = parse_status_content(report);
        validate_layout(&table, Path::new("field report"));
        if table.rows.is_empty() || !table.columns.contains_key("Field") {
            return Err("field report must contain a Field column and test rows".into());
        }
        let field: FieldId = table.rows[0]
            .field
            .parse()
            .map_err(|error| format!("unsupported report field: {error}"))?;
        if seen_fields.contains(&field) {
            return Err(format!("duplicate report for {}", field.name()));
        }
        seen_fields.push(field);
        let mut names = HashMap::default();
        for (index, row) in table.rows.iter().enumerate() {
            if row.field != field.name() {
                return Err("each input report must contain exactly one field".into());
            }
            if names.insert(row.name.clone(), index).is_some() {
                return Err(format!("duplicate test row: {}", row.name));
            }
            let steps = STEP_KEYS
                .iter()
                .zip(REGRESSION_COLS)
                .map(|(&key, &column)| {
                    let status = match table.cell(row, column).unwrap() {
                        "✅" => Status::Pass,
                        "❌" => Status::Fail,
                        "💥" => Status::Crash,
                        "➖" => Status::Skip,
                        "N/A" => Status::NotApplicable,
                        other => return Err(format!("invalid status {other:?} for {column}")),
                    };
                    Ok((key.to_string(), status))
                })
                .collect::<Result<_, String>>()?;
            results.push(TestResult {
                name: row.name.clone(),
                field,
                steps,
                rows: row.rows,
                cols: row.cols,
                binary_bytes: row.size_bytes,
                vm_debug_bytes: row.vm_debug_size_bytes,
                wasm_bytes: row.wasm_size_bytes,
                wasm_debug_bytes: row.wasm_debug_size_bytes,
                analysis_stats: None,
            });
        }
        if report_index == 0 {
            order = names;
        } else if names.len() != order.len() || names.keys().any(|name| !order.contains_key(name)) {
            return Err("field reports contain different test inventories".into());
        }
    }
    results.sort_by_key(|result| (order[&result.name], result.field.code()));
    Ok(render_markdown(&results))
}

/// The stage columns compared by the regression/growth checks, by header name.
pub(crate) const REGRESSION_COLS: &[&str] = &[
    "Compiled",
    "R1CS",
    "Compile",
    "Witgen Run VM",
    "Witgen Correct",
    "Witgen No Leak",
    "AD Run VM",
    "AD Correct",
    "AD No Leak",
    "WASM Compile",
    "Witgen WASM Run",
    "Witgen WASM Correct",
    "Witgen WASM No Leak",
    "AD WASM Run",
    "AD WASM Correct",
    "AD WASM No Leak",
];

/// Verify the table's header carries every column the checks compare. Columns are resolved by name,
/// so differing layouts still align; a stage column going missing (a renamed header, say) must not
/// pass silently, as it would drop that stage from regression checking.
pub(crate) fn validate_layout(table: &StatusTable, path: &Path) {
    for name in REGRESSION_COLS.iter().chain(&["Test", "Rows", "Cols"]) {
        assert!(
            table.columns.contains_key(*name),
            "Status table {} has an unexpected layout: no {name:?} column. Regenerate it — a \
             misnamed or missing stage column would silently drop that stage from the \
             regression/growth checks.",
            path.display(),
        );
    }
}

/// A cell counts as "handled" if it passed (✅) or the stage legitimately did not apply (N/A).
/// N/A is only ever assigned where a stage genuinely cannot run — the witgen/AD/WASM stages of a
/// program correctly rejected at compile or R1CS time, or the correctness/leak checks that need a
/// witness an expected-failure test never produces. A real failure is always ❌/💥/➖, never N/A,
/// so counting N/A as handled cannot mask a regression. Treating the two alike lets a
/// correctly-rejected expected-failure test score 100% with its full weight instead of a single
/// green cell among thousands.
pub(crate) fn cell_is_handled(cell: &str) -> bool {
    cell == "✅" || cell == "N/A"
}

/// Ignored tests are rendered with every stage set to `➖`. A real failure always records at least
/// the stage that failed or crashed, so an all-skip row is the report's unambiguous ignore marker.
pub(crate) fn row_is_ignored(table: &StatusTable, row: &ParsedRow) -> bool {
    REGRESSION_COLS
        .iter()
        .all(|column| table.cell(row, column) == Some("➖"))
}

pub(crate) fn check_regression(baseline_path: &Path, current_path: &Path) -> i32 {
    let baseline = parse_status_table(baseline_path);
    let current = parse_status_table(current_path);
    validate_layout(&baseline, baseline_path);
    validate_layout(&current, current_path);

    let base_map: HashMap<(&str, &str), &ParsedRow> = baseline
        .rows
        .iter()
        .map(|r| ((r.field.as_str(), r.name.as_str()), r))
        .collect();

    let mut regressions = Vec::new();
    for cur in &current.rows {
        if row_is_ignored(&current, cur) {
            continue;
        }
        let Some(base) = base_map.get(&(cur.field.as_str(), cur.name.as_str())) else {
            continue;
        };
        for &col_name in REGRESSION_COLS {
            // `validate_layout` guarantees both lookups resolve.
            let base_val = baseline.cell(base, col_name).unwrap();
            let cur_val = current.cell(cur, col_name).unwrap();
            // `✅ → N/A` is not a regression: it means the stage stopped being applicable (e.g. a
            // failure test whose rejection now happens earlier), which is still a handled cell.
            if base_val == "✅" && !cell_is_handled(cur_val) {
                regressions.push(format!(
                    "  {} [{}] / {}: ✅ → {}",
                    cur.name, cur.field, col_name, cur_val
                ));
            }
        }
    }

    if regressions.is_empty() {
        eprintln!("No regressions detected.");
        0
    } else {
        eprintln!("REGRESSIONS DETECTED:");
        for r in &regressions {
            eprintln!("{r}");
        }
        1
    }
}

pub(crate) fn check_growth(baseline_path: &Path, current_path: &Path) -> GrowthReport {
    let baseline = parse_status_table(baseline_path);
    let current = parse_status_table(current_path);
    validate_layout(&baseline, baseline_path);
    validate_layout(&current, current_path);

    let mut report = GrowthReport::default();
    for field in FieldId::ALL {
        let name = field.name();
        if current.rows.iter().any(|row| row.field == name) {
            report.blocks.push(ReportBlock::Heading(name.to_string()));
            check_growth_for_field(&baseline, &current, name, &mut report);
        }
    }
    report
}

pub(crate) fn check_growth_for_field(
    baseline: &StatusTable,
    current: &StatusTable,
    field: &str,
    report: &mut GrowthReport,
) {
    let base_map: HashMap<(&str, &str), &ParsedRow> = baseline
        .rows
        .iter()
        .map(|r| ((r.field.as_str(), r.name.as_str()), r))
        .collect();

    // Track stats for existing tests (tests in both baseline and current)
    let mut new_checkmarks: Vec<(String, &str)> = Vec::new(); // (test_name, col_name)
    let mut existing_baseline_checkmarks = 0usize;
    let mut existing_current_checkmarks = 0usize;
    let mut existing_total = 0usize;

    // Track stats for all current tests (including new ones)
    let mut total_current_checkmarks = 0usize;
    let mut total_current_cells = 0usize;

    // Tests that compiled at the baseline and are now refused as expected: `Compiled` passes and
    // every later stage is N/A, so neither the regression check nor the cell counts show them. A
    // refusal that was not expected, or a failure at R1CS, already shows as a failed cell.
    let mut newly_refused: Vec<String> = Vec::new();

    // Size/shape changes, grouped per metric so each metric renders its own folding table.
    // Decreases are good news (improvements); increases are warnings (growth).
    let mut constraint_changes = MetricChanges::new("Constraints");
    let mut witness_changes = MetricChanges::new("Witnesses");
    let mut bytecode_changes = MetricChanges::new("Bytecode Size (bytes)");
    let mut vm_debug_changes = MetricChanges::new("VM Debug Sidecar Size (bytes)");
    let mut wasm_changes = MetricChanges::new("WASM Size (bytes)");
    let mut wasm_debug_changes = MetricChanges::new("WASM Debug Sidecar Size (bytes)");

    for cur in current.rows.iter().filter(|row| row.field == field) {
        // An ignored test is outside the comparison rather than sixteen failed stages. Excluding
        // the row also keeps it from diluting the success-rate summary.
        if row_is_ignored(current, cur) {
            continue;
        }

        // Success rate counts handled cells (✅ or N/A) over every cell. N/A is a legitimate
        // "nothing to run here" outcome, not a gap, so a correctly-rejected expected-failure test
        // scores 100% across its whole row rather than one green cell diluted among thousands.
        for &col_name in REGRESSION_COLS {
            total_current_cells += 1;
            if cell_is_handled(current.cell(cur, col_name).unwrap()) {
                total_current_checkmarks += 1;
            }
        }

        let Some(base) = base_map.get(&(cur.field.as_str(), cur.name.as_str())) else {
            continue;
        };

        // For existing tests: count baseline/current handled cells, and list cells that newly
        // turned green. `new_checkmarks` stays literal-✅ so the report's "turned into
        // checkmarks ✅" list means what it says (a cell that became N/A is handled, but it did
        // not turn green).
        for &col_name in REGRESSION_COLS {
            existing_total += 1;
            let base_val = baseline.cell(base, col_name).unwrap();
            let cur_val = current.cell(cur, col_name).unwrap();
            if cell_is_handled(base_val) {
                existing_baseline_checkmarks += 1;
            }
            if cell_is_handled(cur_val) {
                existing_current_checkmarks += 1;
            }
            if base_val != "✅" && cur_val == "✅" {
                new_checkmarks.push((cur.name.clone(), col_name));
            }
        }

        let refused_as_expected =
            current.cell(cur, "Compiled") == Some("✅") && current.cell(cur, "R1CS") == Some("N/A");
        if base.rows.is_some() && cur.rows.is_none() && refused_as_expected {
            newly_refused.push(format!("- **{}**", cur.name));
        }

        // Record per-metric changes; `MetricChanges::record` sorts each into the decrease
        // (improvement) or increase (warning) bucket for its own table.
        if let (Some(br), Some(cr)) = (base.rows, cur.rows) {
            constraint_changes.record(&cur.name, br, cr);
        }
        if let (Some(bc), Some(cc)) = (base.cols, cur.cols) {
            witness_changes.record(&cur.name, bc, cc);
        }
        if let (Some(bs), Some(cs)) = (base.size_bytes, cur.size_bytes) {
            bytecode_changes.record(&cur.name, bs, cs);
        }
        if let (Some(bs), Some(cs)) = (base.vm_debug_size_bytes, cur.vm_debug_size_bytes) {
            vm_debug_changes.record(&cur.name, bs, cs);
        }
        if let (Some(bw), Some(cw)) = (base.wasm_size_bytes, cur.wasm_size_bytes) {
            wasm_changes.record(&cur.name, bw, cw);
        }
        if let (Some(bw), Some(cw)) = (base.wasm_debug_size_bytes, cur.wasm_debug_size_bytes) {
            wasm_debug_changes.record(&cur.name, bw, cw);
        }
    }

    // Calculate completion percentages
    let existing_baseline_pct = if existing_total > 0 {
        existing_baseline_checkmarks as f64 / existing_total as f64 * 100.0
    } else {
        0.0
    };
    let existing_current_pct = if existing_total > 0 {
        existing_current_checkmarks as f64 / existing_total as f64 * 100.0
    } else {
        0.0
    };
    let existing_pct_change = existing_current_pct - existing_baseline_pct;

    let total_current_pct = if total_current_cells > 0 {
        total_current_checkmarks as f64 / total_current_cells as f64 * 100.0
    } else {
        0.0
    };

    // Always print overall success rate
    report.text(format!(
        "**{field} success rate on test cases: {:.1}%**\n\n",
        total_current_pct
    ));

    // Metrics in display order. Each renders its own folding table, separately for decreases
    // (under Positive Changes) and increases (under Warnings).
    let metrics = [
        &constraint_changes,
        &witness_changes,
        &bytecode_changes,
        &vm_debug_changes,
        &wasm_changes,
        &wasm_debug_changes,
    ];
    let any_decreases = metrics.iter().any(|m| !m.decreases.is_empty());
    let any_increases = metrics.iter().any(|m| !m.increases.is_empty());

    // Print positive news section
    let has_positive_news =
        !new_checkmarks.is_empty() || existing_pct_change > 0.0 || any_decreases;
    if has_positive_news {
        report.text("### Positive Changes\n\n");

        if !new_checkmarks.is_empty() || existing_pct_change.abs() > 0.001 {
            if !new_checkmarks.is_empty() {
                report.blocks.push(ReportBlock::Folded {
                    summary: format!("{} cell(s) turned into checkmarks ✅", new_checkmarks.len()),
                    bold: false,
                    header: String::new(),
                    rows: new_checkmarks
                        .iter()
                        .map(|(test, col)| format!("- **{}** / {}", test, col))
                        .collect(),
                });
            }
            if existing_pct_change.abs() > 0.001 {
                report.text(format!(
                    "- Existing tests: {:.1}% → {:.1}% ({:+.1}%)\n",
                    existing_baseline_pct, existing_current_pct, existing_pct_change
                ));
            }
            report.text("\n");
        }

        // One folding table per metric that shrank.
        for m in metrics {
            push_change_table(report, &format!("{} decreased", m.name), &m.decreases);
        }
    }

    if !any_increases && newly_refused.is_empty() {
        if !has_positive_news {
            report.text("No test improvements or size changes detected.\n");
        } else {
            report.text("No size growth detected.\n");
        }
        return;
    }

    // Print warnings section: tests that stopped compiling, then one folding table per metric
    // that grew.
    report.text("### Warnings\n\n");
    if !newly_refused.is_empty() {
        report.blocks.push(ReportBlock::Folded {
            summary: format!(
                "{} test(s) compiled at the baseline and are now refused",
                newly_refused.len()
            ),
            bold: true,
            header: String::new(),
            rows: newly_refused,
        });
        report.text("\n");
    }
    for m in metrics {
        push_change_table(report, &format!("{} grew", m.name), &m.increases);
    }
}

/// Add a folding section wrapping one before/after table. A no-op when there are no rows, so
/// metrics that did not change don't clutter the report.
pub(crate) fn push_change_table(report: &mut GrowthReport, summary: &str, rows: &[String]) {
    if rows.is_empty() {
        return;
    }
    report.blocks.push(ReportBlock::Folded {
        summary: format!("{summary} ({} test(s))", rows.len()),
        bold: true,
        header: "| Test | Before | After | Change |\n|------|--------|-------|--------|\n"
            .to_string(),
        rows: rows.to_vec(),
    });
}

/// The most bytes of report one PR comment carries. GitHub refuses a comment over 65,536
/// characters, and a character is at least a byte; the margin leaves room for the part line and
/// the comparison details the workflow appends.
pub(crate) const GROWTH_COMMENT_BYTES: usize = 60_000;

/// The growth report, as blocks a PR comment can be split between.
#[derive(Default)]
pub(crate) struct GrowthReport {
    pub(crate) blocks: Vec<ReportBlock>,
}

pub(crate) enum ReportBlock {
    /// A field's section heading, repeated at the top of a comment that continues its section.
    Heading(String),
    /// Text kept whole.
    Text(String),
    /// A folding list, one row per line, which may be split between comments.
    Folded {
        summary: String,
        bold: bool,
        /// Lines printed before the rows, such as a table's header.
        header: String,
        rows: Vec<String>,
    },
}

impl GrowthReport {
    pub(crate) fn text(&mut self, text: impl Into<String>) {
        self.blocks.push(ReportBlock::Text(text.into()));
    }

    /// The whole report.
    pub(crate) fn render(&self) -> String {
        self.blocks
            .iter()
            .map(|block| match block {
                ReportBlock::Heading(name) => render_heading(name, false),
                ReportBlock::Text(text) => text.clone(),
                ReportBlock::Folded {
                    summary,
                    bold,
                    header,
                    rows,
                } => render_folded(summary, *bold, false, header, rows),
            })
            .collect()
    }

    /// The report split into comments of at most `limit` bytes, each numbered when there are
    /// several. Blocks stay whole where they fit; a list too long for one comment continues in the
    /// next, with its summary and header repeated, and the comment repeats the section heading.
    pub(crate) fn into_comments(self, limit: usize) -> Vec<String> {
        // Each part line is under this many bytes, so the pieces are cut to leave room for it.
        const PART_LINE_BYTES: usize = 64;
        let mut comments = Comments {
            budget: limit - PART_LINE_BYTES,
            done: vec![String::new()],
            heading: None,
        };

        for block in self.blocks {
            match block {
                ReportBlock::Heading(name) => {
                    comments.heading = None;
                    comments.push(&render_heading(&name, false));
                    comments.heading = Some(name);
                }
                ReportBlock::Text(text) => comments.push(&text),
                ReportBlock::Folded {
                    summary,
                    bold,
                    header,
                    rows,
                } => {
                    // A piece fits a comment that opens with the continued heading.
                    let room = comments.budget - comments.continued_heading().len();
                    let empty = render_folded(&summary, bold, true, &header, &[]).len();
                    let mut piece: Vec<String> = Vec::new();
                    let mut piece_len = empty;
                    let mut continued = false;
                    for row in rows {
                        if !piece.is_empty() && piece_len + row.len() + 1 > room {
                            let rows = std::mem::take(&mut piece);
                            comments
                                .push(&render_folded(&summary, bold, continued, &header, &rows));
                            piece_len = empty;
                            continued = true;
                        }
                        piece_len += row.len() + 1;
                        piece.push(row);
                    }
                    comments.push(&render_folded(&summary, bold, continued, &header, &piece));
                }
            }
        }

        let mut comments = comments.done;
        let count = comments.len();
        if count > 1 {
            for (index, comment) in comments.iter_mut().enumerate() {
                comment.insert_str(
                    0,
                    &format!("_Growth report, part {} of {count}._\n\n", index + 1),
                );
            }
        }
        comments
    }
}

/// The comments [`GrowthReport::into_comments`] has filled so far.
pub(crate) struct Comments {
    pub(crate) budget: usize,
    /// Never empty: the last is the comment being filled.
    pub(crate) done: Vec<String>,
    /// The section being filled, which a new comment reopens.
    pub(crate) heading: Option<String>,
}

impl Comments {
    pub(crate) fn continued_heading(&self) -> String {
        self.heading
            .as_ref()
            .map_or_else(String::new, |name| render_heading(name, true))
    }

    /// Append `piece`, starting a new comment if it would overfill this one.
    pub(crate) fn push(&mut self, piece: &str) {
        let current = self.done.last().expect("there is always a comment");
        if !current.is_empty() && current.len() + piece.len() > self.budget {
            self.done.push(self.continued_heading());
        }
        self.done
            .last_mut()
            .expect("there is always a comment")
            .push_str(piece);
    }
}

pub(crate) fn render_heading(name: &str, continued: bool) -> String {
    if continued {
        format!("## {name} (continued)\n\n")
    } else {
        format!("## {name}\n\n")
    }
}

pub(crate) fn render_folded(
    summary: &str,
    bold: bool,
    continued: bool,
    header: &str,
    rows: &[String],
) -> String {
    let summary = if continued {
        format!("{summary}, continued")
    } else {
        summary.to_string()
    };
    let summary = if bold {
        format!("<b>{summary}</b>")
    } else {
        summary
    };
    let mut folded = format!("<details>\n<summary>{summary}</summary>\n\n{header}");
    for row in rows {
        folded.push_str(row);
        folded.push('\n');
    }
    folded.push_str("\n</details>\n\n");
    folded
}

/// Per-metric before/after changes for the growth report, split into shrink (`decreases`,
/// reported as improvements) and grow (`increases`, reported as warnings) buckets so each
/// metric can render its own folding table.
pub(crate) struct MetricChanges {
    pub(crate) name: &'static str,
    pub(crate) decreases: Vec<String>,
    pub(crate) increases: Vec<String>,
}

impl MetricChanges {
    pub(crate) fn new(name: &'static str) -> Self {
        MetricChanges {
            name,
            decreases: Vec::new(),
            increases: Vec::new(),
        }
    }

    /// Bucket a single test's change as a pre-formatted table row.
    pub(crate) fn record(&mut self, test: &str, before: usize, after: usize) {
        // A metric appearing from zero has no meaningful percentage.
        let pct = if before == 0 {
            "new".to_string()
        } else {
            let pct = (after as f64 - before as f64) / before as f64 * 100.0;
            format!("{pct:+.1}%")
        };
        if after > before {
            self.increases.push(format!(
                "| {} | {} | {} | +{} ({}) |",
                test,
                before,
                after,
                after - before,
                pct
            ));
        } else if after < before {
            self.decreases.push(format!(
                "| {} | {} | {} | {} ({}) |",
                test,
                before,
                after,
                after as i64 - before as i64,
                pct
            ));
        }
    }
}

// The program is a single artifact with both the witgen and AD entry points, so each target has
// one executable size, one debug-sidecar size, and one compile column. The run/correctness/leak
// columns stay split because witgen and AD are distinct executions of their respective entry
// points.
pub(crate) fn render_markdown(results: &[TestResult]) -> String {
    let mut md = String::new();
    md.push_str(&format!("| {} |\n", STATUS_COLUMNS.join(" | ")));
    md.push_str(&format!(
        "| {} |\n",
        vec!["---"; STATUS_COLUMNS.len()].join(" | ")
    ));

    for r in results {
        let s = |key: &str| r.steps.get(key).copied().unwrap_or(Status::Skip).emoji();
        let rows = r.rows.map_or("-".to_string(), |v| v.to_string());
        let cols = r.cols.map_or("-".to_string(), |v| v.to_string());
        let size = r.binary_bytes.map_or("-".to_string(), |v| v.to_string());
        let vm_debug_size = r.vm_debug_bytes.map_or("-".to_string(), |v| v.to_string());
        let wasm_size = r.wasm_bytes.map_or("-".to_string(), |v| v.to_string());
        let wasm_debug_size = r
            .wasm_debug_bytes
            .map_or("-".to_string(), |v| v.to_string());
        md.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            r.name,
            s("COMPILED"),
            s("R1CS"),
            rows,
            cols,
            size,
            vm_debug_size,
            wasm_size,
            wasm_debug_size,
            s("COMPILE"),
            s("WITGEN_RUN"),
            s("WITGEN_CORRECT"),
            s("WITGEN_NOLEAK"),
            s("AD_RUN"),
            s("AD_CORRECT"),
            s("AD_NOLEAK"),
            s("WASM_COMPILE"),
            s("WITGEN_WASM_RUN"),
            s("WITGEN_WASM_CORRECT"),
            s("WITGEN_WASM_NOLEAK"),
            s("AD_WASM_RUN"),
            s("AD_WASM_CORRECT"),
            s("AD_WASM_NOLEAK"),
            r.field.name(),
        ));
    }

    md.push('\n');
    md.push_str(
        "Success rates count passed or not-applicable stages; ignored tests are excluded.\n\n",
    );
    for field in FieldId::ALL {
        if !results.iter().any(|result| result.field == field) {
            continue;
        }
        let (handled, total) = success_counts(results, field);
        let percent = if total == 0 {
            0.0
        } else {
            100.0 * handled as f64 / total as f64
        };
        md.push_str(&format!(
            "- **{} success rate: {percent:.1}%** ({handled}/{total} stages)\n",
            field.name()
        ));
    }
    md
}

pub(crate) fn success_counts(results: &[TestResult], field: FieldId) -> (usize, usize) {
    let mut handled = 0;
    let mut total = 0;
    for result in results.iter().filter(|result| result.field == field) {
        if result.steps.values().all(|status| *status == Status::Skip) {
            continue;
        }
        for &key in STEP_KEYS {
            total += 1;
            if matches!(
                result.steps.get(key),
                Some(Status::Pass | Status::NotApplicable)
            ) {
                handled += 1;
            }
        }
    }
    (handled, total)
}
