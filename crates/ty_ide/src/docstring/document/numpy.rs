use std::borrow::Cow;

use indexmap::IndexMap;
use ruff_text_size::{TextRange, TextSize};

use super::preformatted::{PreformattedBlockScanner, starts_preformatted_block};
use super::syntax::{
    ParsedLine, container_block_end, is_dotted_identifier, is_markdown_code_span, parsed_lines,
    split_once_at_top_level_colon, starts_container_block,
};
use super::{DescriptionBuilder, HeaderKind, SectionKind};

/// Returns parameter documentation from recognized NumPy-style parameter sections.
///
/// `normalized_source` must have already undergone PEP-257 trimming and universal newline
/// normalization.
pub(super) fn parameter_documentation(normalized_source: &str) -> IndexMap<String, String> {
    let mut parameters = Parameters::default();

    for section in sections(normalized_source) {
        let Section {
            kind,
            range: _,
            body,
        } = section;
        if matches!(kind, SectionKind::Parameters | SectionKind::OtherParameters) {
            parameters.extend_fragments(body.into_fragments());
        }
    }

    parameters.into_inner()
}

/// Returns recognized NumPy-style sections in source order.
///
/// `source` must have already undergone PEP-257 trimming and universal newline normalization
/// (typically via `docstring::documentation_trim`).
pub(in crate::docstring) fn sections(source: &str) -> impl Iterator<Item = Section> {
    Parser::new(parsed_lines(source)).parse().into_iter()
}

/// A recognized NumPy-style docstring section.
pub(in crate::docstring) type Section = super::Section<Option<String>>;

type SectionBody = super::SectionBody<Option<String>>;

/// One parsed fragment in a NumPy section body.
pub(in crate::docstring) type BodyFragment = super::BodyFragment<Option<String>>;

/// A named or anonymous item in a NumPy section.
pub(in crate::docstring) type Item = super::Item<Option<String>>;

struct Parser<'a> {
    lines: Vec<ParsedLine<'a>>,
    current_line: usize,
    sections: Vec<Section>,
    current_section: Option<SectionBuilder<'a>>,
    scanner: PreformattedBlockScanner<'a>,
}

impl<'a> Parser<'a> {
    fn new(lines: Vec<ParsedLine<'a>>) -> Self {
        Self {
            lines,
            current_line: 0,
            sections: Vec::new(),
            current_section: None,
            scanner: PreformattedBlockScanner::default(),
        }
    }

    fn parse(mut self) -> Vec<Section> {
        while self.current_line < self.lines.len() {
            self.push_line();
        }

        if let Some(section) = self.current_section.take() {
            self.finish_section(section);
        }

        self.sections
    }

    fn push_line(&mut self) {
        let line = self.lines[self.current_line];
        let line_header = self.parse_header(self.current_line);
        let index = self.current_line;
        self.current_line += 1;

        // First, attempt to add the current line to the current section.
        if let Some(mut section) = self.current_section.take() {
            if section.push_line(line, line_header, &self.lines[self.current_line..]) {
                self.current_section = Some(section);
                return;
            }

            self.finish_section(section);
        }

        // Second, skip content owned by a preformatted or container block, where nested headers
        // are inert.
        if self.scanner.consume_preformatted_line(line.text) {
            return;
        }
        if let Some(end) = container_block_end(&self.lines, index) {
            self.current_line = end;
            return;
        }

        // Finally, start a new section from a standalone header, or observe syntax that may
        // introduce a preformatted block.
        if let Some(header) = line_header {
            self.current_section = Some(SectionBuilder::new(header));
            self.current_line += 1;
        } else {
            self.scanner
                .observe_line_outside_preformatted_block(line.text);
        }
    }

    fn parse_header(&self, index: usize) -> Option<Header> {
        let line = self.lines[index];
        let underline = self.lines.get(index + 1)?;

        if line.text.trim().is_empty() || !is_underline(underline.text) {
            return None;
        }

        let indent = if index == 0 {
            // PEP 257 trimming strips the indentation from the first line,
            // so instead use the underline to determine this section's indentation.
            underline.indent
        } else if underline.indent == line.indent {
            line.indent
        } else {
            // After the first line, each underline must align with its section title.
            return None;
        };

        Some(Header {
            kind: section_kind(line.text)
                .map(HeaderKind::Structured)
                .unwrap_or(HeaderKind::Opaque),
            indent,
            range: TextRange::new(line.range.start(), underline.range.end()),
        })
    }

    fn finish_section(&mut self, section: SectionBuilder<'a>) {
        if let Some(section) = section.finish() {
            self.sections.push(section);
        }
    }
}

struct SectionBuilder<'a> {
    section_header: Header,
    range: TextRange,
    /// Blank lines whose ownership depends on the next nonblank line.
    pending_blank_lines: Vec<ParsedLine<'a>>,
    /// Prevents code examples from participating in section-boundary detection.
    preformatted: PreformattedBlockScanner<'a>,
    /// Whether a confirmed item aligned with the section heading has been seen.
    has_item: bool,
    body: BodyBuilder<'a>,
}

impl<'a> SectionBuilder<'a> {
    fn new(section_header: Header) -> Self {
        Self {
            range: section_header.range,
            pending_blank_lines: Vec::new(),
            preformatted: PreformattedBlockScanner::default(),
            has_item: false,
            body: BodyBuilder::new(section_header.kind, section_header.indent),
            section_header,
        }
    }

    /// Returns `false` when `line` belongs outside this section.
    fn push_line(
        &mut self,
        line: ParsedLine<'a>,
        line_header: Option<Header>,
        following_lines: &[ParsedLine<'_>],
    ) -> bool {
        // First, let an active preformatted block consume the line before interpreting it.
        let preformatted_block_is_active = self.preformatted.is_active();
        let line_is_preformatted = self.preformatted.consume_preformatted_line(line.text);
        if preformatted_block_is_active && line_is_preformatted {
            self.commit_pending_blank_lines();
            self.push_content_line(line, ItemLine::default());
            return true;
        }

        // Second, defer blank lines until the next content line determines their ownership.
        if line.text.trim().is_empty() {
            self.pending_blank_lines.push(line);
            return true;
        }

        // Third, classify a nonblank line and stop if it begins content outside this section.
        let item_line = ItemLine::classify(self.section_header.kind, line, following_lines);
        let is_top_level_item =
            item_line.confirmed_item.is_some() && line.indent == self.section_header.indent;
        let has_leading_blank_lines = !self.pending_blank_lines.is_empty();
        if self.should_end_before(
            line,
            line_header,
            is_top_level_item,
            has_leading_blank_lines,
        ) {
            return false;
        }

        // Finally, commit the accepted line and update the state used to classify later lines.
        self.commit_pending_blank_lines();
        self.push_content_line(line, item_line);
        if is_top_level_item {
            self.has_item = true;
        }
        if !line_is_preformatted {
            self.preformatted
                .observe_line_outside_preformatted_block(line.text);
        }

        true
    }

    fn should_end_before(
        &self,
        line: ParsedLine<'_>,
        line_header: Option<Header>,
        is_confirmed_item: bool,
        has_leading_blank_lines: bool,
    ) -> bool {
        // A sibling-level underlined header starts a new section.
        // Every section, including an opaque one, ends at a sibling or shallower header.
        if line_header.is_some_and(|header| header.indent <= self.section_header.indent) {
            return true;
        }

        // Items are not parsed in opaque sections so only the above header can end them.
        if self.section_header.kind == HeaderKind::Opaque {
            return false;
        }

        match line.indent.cmp(&self.section_header.indent) {
            std::cmp::Ordering::Less => true,
            std::cmp::Ordering::Greater => false,
            std::cmp::Ordering::Equal => {
                let is_aligned_non_item = !is_confirmed_item;

                if self.section_header.kind.is_parameter_section() {
                    // Parameter sections may contain leading prose and aligned continuations.
                    // After an item establishes the list, a blank line followed by an aligned
                    // non-item ends the section.
                    self.has_item && has_leading_blank_lines && is_aligned_non_item
                } else {
                    is_aligned_non_item
                }
            }
        }
    }

    fn commit_pending_blank_lines(&mut self) {
        let lines = std::mem::take(&mut self.pending_blank_lines);
        for line in lines {
            self.range = self.range.cover(line.range);
            self.body.push_blank_line();
        }
    }

    fn push_content_line(&mut self, line: ParsedLine<'a>, item_line: ItemLine<'a>) {
        self.range = self.range.cover(line.range);
        self.body.push_line(line, item_line);
    }

    fn finish(self) -> Option<Section> {
        let HeaderKind::Structured(kind) = self.section_header.kind else {
            return None;
        };

        Some(Section {
            kind,
            range: self.range,
            body: self.body.finish(),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Header {
    kind: HeaderKind,
    indent: TextSize,
    range: TextRange,
}

fn section_kind(line: &str) -> Option<SectionKind> {
    match line.trim().to_ascii_lowercase().as_str() {
        "parameters" => Some(SectionKind::Parameters),
        "other parameters" => Some(SectionKind::OtherParameters),
        "attributes" => Some(SectionKind::Attributes),
        "returns" => Some(SectionKind::Returns),
        "yields" => Some(SectionKind::Yields),
        "raises" => Some(SectionKind::Raises),
        _ => None,
    }
}

fn is_underline(line: &str) -> bool {
    let line = line.trim();
    line.len() >= 3 && line.chars().all(|char| char == '-')
}

/// Accepts description-backed items, plus single-token types without one.
fn is_anonymous_return_item(line: &str, has_description: bool) -> bool {
    !line.is_empty()
        && !line.ends_with('.')
        && !line.ends_with(':')
        && (has_description || !line.chars().any(char::is_whitespace))
}

enum BodyBuilder<'a> {
    /// A recognized section whose body consists of named items and their descriptions.
    ItemList(ItemListBuilder<'a>),
    /// An underlined section that participates in boundary detection but is not parsed.
    Opaque,
}

impl<'a> BodyBuilder<'a> {
    fn new(kind: HeaderKind, required_item_indent: TextSize) -> Self {
        match kind {
            HeaderKind::Structured(_) => Self::ItemList(ItemListBuilder::new(
                kind.is_parameter_section(),
                required_item_indent,
            )),
            HeaderKind::Opaque => Self::Opaque,
        }
    }

    fn push_blank_line(&mut self) {
        if let Self::ItemList(builder) = self {
            builder.push_blank_line();
        }
    }

    fn push_line(&mut self, line: ParsedLine<'a>, item_line: ItemLine<'a>) {
        if let Self::ItemList(builder) = self {
            builder.push_line(line, item_line);
        }
    }

    fn finish(self) -> SectionBody {
        match self {
            Self::ItemList(builder) => builder.finish(),
            Self::Opaque => SectionBody::Opaque,
        }
    }
}

struct ItemListBuilder<'a> {
    fragments: Vec<BodyFragment>,
    current_item: Option<ItemBuilder<'a>>,
    leading_prose: DescriptionBuilder<'a>,
    required_item_indent: TextSize,
    preserve_leading_prose: bool,
    has_structural_ambiguity: bool,
}

impl<'a> ItemListBuilder<'a> {
    fn new(preserve_leading_prose: bool, required_item_indent: TextSize) -> Self {
        Self {
            fragments: Vec::new(),
            current_item: None,
            leading_prose: DescriptionBuilder::default(),
            required_item_indent,
            preserve_leading_prose,
            has_structural_ambiguity: false,
        }
    }

    fn push_blank_line(&mut self) {
        if let Some(item) = &mut self.current_item {
            item.description.push_continuation("");
        } else if self.preserve_leading_prose {
            self.leading_prose.push_continuation("");
        }
    }

    fn push_line(&mut self, line: ParsedLine<'a>, item_line: ItemLine<'a>) {
        let line_indent = line.indent;
        let ItemLine {
            confirmed_item,
            has_structural_ambiguity,
        } = item_line;
        if line_indent == self.required_item_indent {
            if let Some(item) = confirmed_item {
                self.finish_leading_prose();
                self.finish_current_item();
                self.current_item = Some(item);
                self.has_structural_ambiguity |= has_structural_ambiguity;
                return;
            }

            // An unconfirmed item-like line, or aligned prose after an item, remains ordinary
            // content but makes structured rendering ambiguous.
            if has_structural_ambiguity || self.current_item.is_some() {
                self.has_structural_ambiguity = true;
            }
        }
        if self.current_item.is_none()
            && self.preserve_leading_prose
            && line_indent != self.required_item_indent
        {
            // DescriptionBuilder trims prose indentation. Deeper preamble content may instead be
            // nested content or code, so leave rendering raw.
            self.has_structural_ambiguity = true;
        }

        if let Some(item) = &mut self.current_item {
            item.description.push_continuation(line.text);
        } else if self.preserve_leading_prose {
            self.leading_prose.push_line(line.text);
        } else {
            self.has_structural_ambiguity = true;
        }
    }

    fn finish_leading_prose(&mut self) {
        let prose = std::mem::take(&mut self.leading_prose).finish();
        if !prose.is_empty() {
            self.fragments.push(BodyFragment::Prose(prose));
        }
    }

    fn finish_current_item(&mut self) {
        if let Some(item) = self.current_item.take() {
            self.fragments.push(BodyFragment::Item(item.finish()));
        }
    }

    fn finish(mut self) -> SectionBody {
        if self.preserve_leading_prose && self.current_item.is_none() {
            return SectionBody::Opaque;
        }

        self.finish_leading_prose();
        self.finish_current_item();
        SectionBody::Parsed {
            fragments: self.fragments,
            has_structural_ambiguity: self.has_structural_ambiguity,
        }
    }
}

#[derive(Clone, Copy)]
enum NamedSectionKind {
    Parameter,
    Attribute,
}

#[derive(Default)]
struct ItemLine<'a> {
    /// An item header accepted by the section-specific confirmation rules.
    confirmed_item: Option<ItemBuilder<'a>>,
    /// Whether this line makes structured rendering ambiguous at item indentation.
    has_structural_ambiguity: bool,
}

impl<'a> ItemLine<'a> {
    fn classify(
        section_kind: HeaderKind,
        line: ParsedLine<'a>,
        following_lines: &[ParsedLine<'_>],
    ) -> Self {
        // Each structured section has its own item grammar. Opaque sections only delimit content.
        let HeaderKind::Structured(kind) = section_kind else {
            return Self::default();
        };

        match kind {
            SectionKind::Parameters
            | SectionKind::KeywordArguments
            | SectionKind::OtherParameters => {
                Self::classify_named(NamedSectionKind::Parameter, line, following_lines)
            }
            SectionKind::Attributes => {
                Self::classify_named(NamedSectionKind::Attribute, line, following_lines)
            }
            SectionKind::Returns | SectionKind::Yields => {
                Self::classify_return(line, following_lines)
            }
            SectionKind::Raises => Self::classify_raise(line),
        }
    }

    fn classify_named(
        kind: NamedSectionKind,
        line: ParsedLine<'a>,
        following_lines: &[ParsedLine<'_>],
    ) -> Self {
        let trimmed = line.text.trim();

        // Prefer the explicit `name : type` form. Whitespace before the colon or a description
        // block confirms the item; a nonempty parameter type also confirms a compact separator.
        if let Some(separator) = parse_type_separator(trimmed) {
            let has_description_block = has_indented_description(&line, following_lines);
            let is_confirmed_item = separator.has_whitespace_before_colon
                || has_description_block
                || (matches!(kind, NamedSectionKind::Parameter) && !separator.ty.is_empty());

            if !is_confirmed_item {
                return Self::unconfirmed_candidate();
            }
            return Self::confirmed(
                ItemBuilder::new(
                    Some(normalize_item_name(separator.name)),
                    Some(separator.ty),
                    "",
                ),
                separator.has_structural_ambiguity,
            );
        }

        // Parameters may omit their type. A bare attribute needs a description block to
        // distinguish it from prose.
        if !is_item_name(trimmed) {
            return Self::default();
        }
        let is_confirmed_item = matches!(kind, NamedSectionKind::Parameter)
            || has_indented_description(&line, following_lines);
        if !is_confirmed_item {
            return Self::unconfirmed_candidate();
        }
        Self::confirmed(
            ItemBuilder::new(Some(normalize_item_name(trimmed)), None, ""),
            false,
        )
    }

    fn classify_return(line: ParsedLine<'a>, following_lines: &[ParsedLine<'_>]) -> Self {
        let trimmed = line.text.trim();
        // Block openers at item indentation belong outside the section, not to a return item.
        if starts_preformatted_block(trimmed) || starts_container_block(trimmed) {
            return Self::default();
        }

        // A complete code span is an anonymous type even when its contents contain a colon.
        if is_markdown_code_span(trimmed) {
            return Self::confirmed(ItemBuilder::new(None, Some(trimmed), ""), false);
        }

        // Next, prefer the named `name : type` form. A colon adjacent to the name needs a
        // description block to distinguish it from prose.
        let has_description_block = has_indented_description(&line, following_lines);
        if let Some(separator) = parse_return_type_separator(trimmed) {
            let is_confirmed_item = separator.has_whitespace_before_colon || has_description_block;
            if !is_confirmed_item {
                return Self::unconfirmed_candidate();
            }
            return Self::confirmed(
                ItemBuilder::new(Some(Cow::Borrowed(separator.name)), Some(separator.ty), ""),
                separator.has_structural_ambiguity,
            );
        }

        // Finally, accept an anonymous type only when its shape or description distinguishes it
        // from prose.
        if !is_anonymous_return_item(trimmed, has_description_block) {
            return Self::default();
        }
        Self::confirmed(ItemBuilder::new(None, Some(trimmed), ""), false)
    }

    fn classify_raise(line: ParsedLine<'a>) -> Self {
        let trimmed = line.text.trim();
        // Raises use a named item, with an optional inline description after the first colon.
        let (name, description) = trimmed
            .split_once(':')
            .map_or((trimmed, ""), |(name, description)| {
                (name.trim(), description.trim())
            });
        if !is_item_name(name) {
            return Self::default();
        }
        Self::confirmed(
            ItemBuilder::new(Some(Cow::Borrowed(name)), None, description),
            false,
        )
    }

    fn confirmed(confirmed_item: ItemBuilder<'a>, has_structural_ambiguity: bool) -> Self {
        Self {
            confirmed_item: Some(confirmed_item),
            has_structural_ambiguity,
        }
    }

    /// Marks an unconfirmed item-like line as ambiguous without retaining its parsed fields.
    fn unconfirmed_candidate() -> Self {
        Self {
            confirmed_item: None,
            has_structural_ambiguity: true,
        }
    }
}

struct ItemBuilder<'a> {
    display_name: Option<Cow<'a, str>>,
    ty: Option<&'a str>,
    description: DescriptionBuilder<'a>,
}

impl<'a> ItemBuilder<'a> {
    fn new(
        display_name: Option<Cow<'a, str>>,
        ty: Option<&'a str>,
        inline_description: &'a str,
    ) -> Self {
        Self {
            display_name,
            ty,
            description: DescriptionBuilder::with_inline(inline_description),
        }
    }

    fn finish(self) -> Item {
        Item {
            display_name: self.display_name.map(Cow::into_owned),
            ty: self.ty.map(str::to_string),
            description: self.description.finish(),
        }
    }
}

#[derive(Default)]
struct Parameters(IndexMap<String, String>);

impl Parameters {
    fn extend_fragments(&mut self, fragments: Vec<BodyFragment>) {
        for fragment in fragments {
            let BodyFragment::Item(item) = fragment else {
                continue;
            };
            let Item {
                display_name,
                ty: _,
                description,
            } = item;
            let Some(display_name) = display_name else {
                continue;
            };
            let description = description.trim();
            if description.is_empty() {
                continue;
            }
            let Some(names) = parameter_lookup_names(&display_name) else {
                continue;
            };
            for name in names {
                self.0.insert(name, description.to_string());
            }
        }
    }

    fn into_inner(self) -> IndexMap<String, String> {
        self.0
    }
}

fn parameter_lookup_names(display_name: &str) -> Option<Vec<String>> {
    let mut lookup_names = Vec::new();
    for name in display_name.split(',').map(str::trim) {
        if name == "..." {
            continue;
        }

        let name = normalize_item_name(name);
        if !is_item_name_part(&name) {
            return None;
        }
        lookup_names.push(name.into_owned());
    }

    (!lookup_names.is_empty()).then_some(lookup_names)
}

/// A parsed NumPy-style `name : type` separator.
struct TypeSeparator<'a> {
    /// The documented item name.
    name: &'a str,
    /// The documented item type.
    ty: &'a str,
    /// Whether whitespace before the colon identifies conventional NumPy item syntax.
    has_whitespace_before_colon: bool,
    /// Whether the separator omits whitespace on both sides.
    has_structural_ambiguity: bool,
}

/// Parses a NumPy-style `name : type` separator.
fn parse_type_separator(line: &str) -> Option<TypeSeparator<'_>> {
    parse_type_separator_if(line, is_item_name)
}

/// Parses a return separator without interpreting its display name.
fn parse_return_type_separator(line: &str) -> Option<TypeSeparator<'_>> {
    parse_type_separator_if(line, |name| !name.is_empty())
}

fn parse_type_separator_if(
    line: &str,
    is_valid_name: impl FnOnce(&str) -> bool,
) -> Option<TypeSeparator<'_>> {
    let (name, ty) = split_once_at_top_level_colon(line)?;
    let has_whitespace_before_colon = name.chars().last().is_some_and(char::is_whitespace);
    let has_whitespace_after_colon = ty.chars().next().is_some_and(char::is_whitespace);
    let has_structural_ambiguity =
        !has_whitespace_before_colon && !has_whitespace_after_colon && !ty.is_empty();

    let name = name.trim();
    let ty = ty.trim();
    if !is_valid_name(name) {
        return None;
    }
    Some(TypeSeparator {
        name,
        ty,
        has_whitespace_before_colon,
        has_structural_ambiguity,
    })
}

fn has_indented_description(line: &ParsedLine<'_>, following_lines: &[ParsedLine<'_>]) -> bool {
    following_lines
        .iter()
        .find(|line| !line.text.trim().is_empty())
        .is_some_and(|next| next.indent > line.indent)
}

/// Returns whether `name` is a valid NumPy-style item name or comma-separated name list.
fn is_item_name(name: &str) -> bool {
    let mut has_lookup_name = false;
    let valid = name.split(',').all(|part| {
        let part = part.trim();
        if part == "..." {
            return true;
        }

        let part = normalize_item_name(part);
        if is_item_name_part(&part) {
            has_lookup_name = true;
            true
        } else {
            false
        }
    });

    valid && has_lookup_name
}

/// Removes reStructuredText escapes from NumPy variadic parameter names.
fn normalize_item_name(name: &str) -> Cow<'_, str> {
    if name.contains(r"\*") {
        Cow::Owned(name.replace(r"\*", "*"))
    } else {
        Cow::Borrowed(name)
    }
}

fn is_item_name_part(name: &str) -> bool {
    let name = name
        .strip_prefix("**")
        .or_else(|| name.strip_prefix('*'))
        .unwrap_or(name);

    is_dotted_identifier(name)
}

#[cfg(test)]
mod tests {
    use super::{BodyFragment, Item, SectionBody, parameter_documentation, sections};

    #[test]
    fn extracts_supported_numpy_parameter_items() {
        let source = normalized(
            r#"
        This is a function description.

        Parameters
        ----------
        param1 : str
            The first parameter description
        param2, param4 : int
            The shared parameter description

            This is a second paragraph.
            This is a continuation of the shared description.
        param3
            A parameter without type annotation
        *args : object
            Extra positional arguments
        **kwargs : object
            Extra keyword arguments
        options.mode : str
            Nested field documentation
        π : int
            A Unicode parameter
        a1, a2, ... : sequence of array_like
            Arrays to combine
        \*escaped_args : object
            Escaped positional arguments
        \**escaped_kwargs : object
            Escaped keyword arguments
        override_repr: callable, optional
            Replacement representation function
        formats, names :
        undocumented
        copy : bool
            Whether to copy the input

        Other Parameters
        ----------------
        kw_only : str, optional
            A less commonly used keyword-only parameter

        Returns
        -------
        str
            The return value description

        Yields
        ------
        int
            The next value
        "#,
        );

        let param_docs = parameter_documentation(&source);

        assert_eq!(param_docs.len(), 15);
        assert_eq!(
            param_docs.get("param1").expect("param1 should exist"),
            "The first parameter description"
        );
        assert_eq!(
            param_docs.get("param2").expect("param2 should exist"),
            "\
The shared parameter description

This is a second paragraph.
This is a continuation of the shared description."
        );
        assert_eq!(
            param_docs.get("param4").expect("param4 should exist"),
            "\
The shared parameter description

This is a second paragraph.
This is a continuation of the shared description."
        );
        assert_eq!(
            param_docs.get("param3").expect("param3 should exist"),
            "A parameter without type annotation"
        );
        assert_eq!(
            param_docs.get("*args").expect("*args should exist"),
            "Extra positional arguments"
        );
        assert_eq!(
            param_docs.get("**kwargs").expect("**kwargs should exist"),
            "Extra keyword arguments"
        );
        assert!(!param_docs.contains_key("options"));
        assert_eq!(
            param_docs
                .get("options.mode")
                .expect("options.mode should exist"),
            "Nested field documentation"
        );
        assert_eq!(
            param_docs.get("π").expect("π should exist"),
            "A Unicode parameter"
        );
        assert_eq!(
            param_docs.get("a1").expect("a1 should exist"),
            "Arrays to combine"
        );
        assert_eq!(
            param_docs.get("a2").expect("a2 should exist"),
            "Arrays to combine"
        );
        assert_eq!(
            param_docs
                .get("*escaped_args")
                .expect("*escaped_args should exist"),
            "Escaped positional arguments"
        );
        assert_eq!(
            param_docs
                .get("**escaped_kwargs")
                .expect("**escaped_kwargs should exist"),
            "Escaped keyword arguments"
        );
        assert_eq!(
            param_docs
                .get("override_repr")
                .expect("override_repr should exist"),
            "Replacement representation function"
        );
        assert_eq!(
            param_docs.get("copy").expect("copy should exist"),
            "Whether to copy the input"
        );
        assert_eq!(
            param_docs.get("kw_only").expect("kw_only should exist"),
            "A less commonly used keyword-only parameter"
        );

        let duplicate_source = normalized(
            r#"
        Parameters
        ----------
        value : str
            First documentation.
        value : str
            Replacement documentation.
        "#,
        );

        assert_eq!(
            parameter_documentation(&duplicate_source)["value"],
            "Replacement documentation."
        );
    }

    #[test]
    fn extracts_shifted_top_level_numpy_sections() {
        let source = normalized(
            "\
A decoded newline follows:
This line starts at column zero.

    Parameters
    ----------
    shifted : int
        Documentation in a shifted section.

    Returns
    -------
    bool
        Result.",
        );

        assert_eq!(
            parameter_documentation(&source)["shifted"],
            "Documentation in a shifted section."
        );
    }

    #[test]
    fn ignores_numpy_items_nested_in_section_preambles() {
        let source = normalized(
            "\
Parameters
----------
Choose one of the following.
    nested : int
        Example-only text.
beta : float
    Useful documentation.",
        );

        let parameter_documentation = parameter_documentation(&source);
        assert_eq!(parameter_documentation.len(), 1);
        assert_eq!(parameter_documentation["beta"], "Useful documentation.");
    }

    #[test]
    fn ignores_numpy_sections_in_containers() {
        let raw = "\
Summary.

- Example data:
    Parameters
    ----------
    nested : int
        Not parameter documentation.";
        let source = normalized(raw);

        assert!(parameter_documentation(&source).is_empty(), "{raw}");
    }

    #[test]
    fn ignores_numpy_sections_in_rest_literal_blocks() {
        let source = normalized(
            "\
Summary.

Example::

    Other Parameters
    ----------------
    nested : int
        Literal content.",
        );

        assert!(parameter_documentation(&source).is_empty());
    }

    #[test]
    fn finds_numpy_section_after_first_line_rest_literal_block() {
        let source = normalized(
            "\
Example::

      sample output

    Parameters
    ----------
    value : int
        Parameter documentation.",
        );

        assert_eq!(
            parameter_documentation(&source)["value"],
            "Parameter documentation."
        );
    }

    #[test]
    fn ignores_numpy_sections_nested_in_other_sections() {
        let source = normalized(
            "\
Examples
--------
    Parameters
    ----------
    nested : int
        Not parameter documentation.

Notes
-----
More details.

Parameters
----------
value : int
    Parameter documentation.",
        );

        let parameter_documentation = parameter_documentation(&source);
        assert_eq!(parameter_documentation.len(), 1);
        assert_eq!(parameter_documentation["value"], "Parameter documentation.");
        assert!(!parameter_documentation.contains_key("nested"));
    }

    #[test]
    fn extracts_parameters_from_a_first_line_section() {
        let source = normalized(
            "\
Parameters
    ----------
    value : int
        Description.

Examples:
    Example prose.",
        );
        let documentation = parameter_documentation(&source);

        assert_eq!(documentation["value"], "Description.");
        assert!(
            sections(&source)
                .next()
                .is_some_and(|section| &source[section.range]
                    == "\
Parameters
    ----------
    value : int
        Description.")
        );
    }

    #[test]
    fn preserves_blank_lines_in_preformatted_parameter_descriptions() {
        let source = normalized(
            "\
Parameters
----------
value : str
    ```text
    first

    second
    ```
other : int
    Another value.",
        );
        let documentation = parameter_documentation(&source);

        assert_eq!(
            documentation["value"],
            "\
```text
first

second
```"
        );
        assert_eq!(documentation["other"], "Another value.");
    }

    #[test]
    fn ignores_parameter_items_not_aligned_with_section_heading() {
        let source = normalized(
            "\
Parameters
----------
    value : int
        Description.
    other : str
        Other.",
        );

        assert!(parameter_documentation(&source).is_empty());
        assert!(
            sections(&source)
                .next()
                .is_some_and(|section| matches!(section.body, SectionBody::Opaque))
        );
    }

    #[test]
    fn extracts_compact_parameters_without_rendering_them_structurally() {
        let source = normalized(
            "\
Parameters
----------
d:int
    Parameter d.",
        );
        let documentation = parameter_documentation(&source);

        assert_eq!(documentation["d"], "Parameter d.");
        assert!(sections(&source).next().is_some_and(|section| matches!(
            section.body,
            SectionBody::Parsed {
                has_structural_ambiguity: true,
                ..
            }
        )));
    }

    #[test]
    fn leaves_indented_parameter_preambles_raw() {
        let source = normalized(
            "\
Parameters
----------
Choose one form.
    foo()
beta : int
    Useful documentation.",
        );
        let documentation = parameter_documentation(&source);

        assert_eq!(documentation["beta"], "Useful documentation.");
        assert!(sections(&source).next().is_some_and(|section| matches!(
            section.body,
            SectionBody::Parsed {
                has_structural_ambiguity: true,
                ..
            }
        )));
    }

    #[test]
    fn leaves_unconfirmed_parameter_item_opaque() {
        let source = normalized(
            "\
Summary.

Parameters
----------
Note:",
        );
        assert!(
            sections(&source)
                .next()
                .is_some_and(|section| matches!(section.body, SectionBody::Opaque))
        );
    }

    #[test]
    fn extracts_later_parameters_from_an_ambiguous_section() {
        let source = normalized(
            "\
Parameters
----------
value : int
    Description.
Ambiguous prose.
other : str
    Other.",
        );
        let documentation = parameter_documentation(&source);

        assert_eq!(
            documentation["value"],
            "\
Description.
Ambiguous prose."
        );
        assert_eq!(documentation["other"], "Other.");
        assert!(sections(&source).next().is_some_and(|section| matches!(
            section.body,
            SectionBody::Parsed {
                has_structural_ambiguity: true,
                ..
            }
        )));
    }

    #[test]
    fn extracts_later_aligned_parameters_from_an_ambiguous_section() {
        let source = normalized(
            "\
Parameters
----------
value : int
    Description.
malformed name : str
other : str
    Other.",
        );
        let documentation = parameter_documentation(&source);

        assert_eq!(
            documentation["value"],
            "\
Description.
malformed name : str"
        );
        assert_eq!(documentation["other"], "Other.");
        assert!(sections(&source).next().is_some_and(|section| matches!(
            section.body,
            SectionBody::Parsed {
                has_structural_ambiguity: true,
                ..
            }
        )));
    }

    #[test]
    fn extracts_later_parameters_after_an_unconfirmed_item() {
        let source = normalized(
            "\
Parameters
----------
value : int
    Description.
Note:
other : str
    Other.",
        );
        let documentation = parameter_documentation(&source);

        assert_eq!(
            documentation["value"],
            "\
Description.
Note:"
        );
        assert_eq!(documentation["other"], "Other.");
        assert!(sections(&source).next().is_some_and(|section| matches!(
            section.body,
            SectionBody::Parsed {
                has_structural_ambiguity: true,
                ..
            }
        )));
    }

    #[test]
    fn treats_indented_return_items_as_structurally_ambiguous() {
        let source = normalized(
            "\
Returns
-------
    foo()
    result",
        );

        assert!(sections(&source).next().is_some_and(|section| matches!(
            section.body,
            SectionBody::Parsed {
                has_structural_ambiguity: true,
                ..
            }
        )));
    }

    #[test]
    fn ends_parameters_before_preformatted_block() {
        let source = normalized(
            "\
Parameters
----------
value : int
    Description.

```text
other : str
```",
        );

        assert!(sections(&source).next().is_some_and(|section| {
            matches!(
                section.body,
                SectionBody::Parsed {
                    has_structural_ambiguity: false,
                    ..
                }
            ) && &source[section.range]
                == "\
Parameters
----------
value : int
    Description."
        }));
    }

    #[test]
    fn parses_return_items_from_structure() {
        let source = normalized(
            "\
Returns
-------
np.ndarray, bool
    The values and a flag.
angular separation : Quantity
    The angle between two points.
`module:Type`",
        );

        assert_eq!(
            sections(&source).next().map(|section| section.body),
            Some(SectionBody::Parsed {
                fragments: vec![
                    BodyFragment::Item(Item {
                        display_name: None,
                        ty: Some("np.ndarray, bool".to_string()),
                        description: "The values and a flag.".to_string(),
                    }),
                    BodyFragment::Item(Item {
                        display_name: Some("angular separation".to_string()),
                        ty: Some("Quantity".to_string()),
                        description: "The angle between two points.".to_string(),
                    }),
                    BodyFragment::Item(Item {
                        display_name: None,
                        ty: Some("`module:Type`".to_string()),
                        description: String::new(),
                    }),
                ],
                has_structural_ambiguity: false,
            })
        );
    }

    fn normalized(raw: &str) -> String {
        crate::docstring::documentation_trim(raw)
    }
}
