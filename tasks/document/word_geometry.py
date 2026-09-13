"""Word-level geometric features for document classification.

Why this module exists
----------------------
The layout detector emits two things: regions and words. Only regions were
being consumed, and at region granularity a whole form line collapses into one
``Text`` block — ``"Incumbent: Y (Y or N)"`` is a single region. The
label-to-value relation, which is the defining structure of a form, is not
representable at that granularity and therefore could not be tested by any
rule. The same loss hides the header block of a memorandum, the justified
right edge of a printed article, and the tab-stop grid of a table of entries.

Word boxes restore that structure. The features here describe *how text is
arranged on the page*, independent of what it says, which is exactly the
evidence that separates the document families whose vocabularies overlap or
whose OCR is too noisy to match.

Input contract
--------------
``page_words`` is whatever the layout stage produced, one entry per page. The
parser is deliberately tolerant of shape — nested per-page lists, dicts with a
``words`` key, ``bbox``/``box``/``geometry`` under several spellings, corner
pairs or flat quadruples, absolute or unit-normalised coordinates — because the
extractor must not be coupled to one detector's serialisation. What it does
require is that coordinates share the coordinate system of ``page_sizes``.
Anything it cannot parse yields no geometry rather than wrong geometry: the
caller then sees the channel as unavailable instead of silently scoring on
noise.
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Iterable, NamedTuple


class Word(NamedTuple):
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    confidence: float | None

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def y_center(self) -> float:
        return (self.y0 + self.y1) / 2.0


_TEXT_KEYS = ("text", "value", "word", "content")
_BOX_KEYS = ("bbox", "box", "geometry", "coordinates", "coords", "rect")
_DEFAULT_GEOMETRY_FEATURES = {
    "geometry_pages": 0,
    "geometry_page_ratio": 0.0,
    "word_line_count": 0,
    "word_token_count": 0,
    "alignment_column_count": 0,
    "tab_stop_count": 0,
    "label_value_line_count": 0,
    "label_value_line_ratio": 0.0,
    "wide_gap_line_ratio": 0.0,
    "body_wide_gap_line_ratio": 0.0,
    "right_edge_regularity": 0.0,
    "line_pitch_regularity": 0.0,
    "indent_ratio": 0.0,
    "short_line_ratio": 0.0,
    "narrative_line_ratio": 0.0,
    "narrative_column_count": 0,
    "newspaper_column_pages": 0,
    "newspaper_column_geometry_ratio": 0.0,
    "centered_line_ratio": 0.0,
    "top_band_header_ratio": 0.0,
    "space_width": 0.0,
    "uppercase_word_ratio": 0.0,
    "word_confidence_mean": None,
}


def _coerce_box(value: Any) -> tuple[float, float, float, float] | None:
    """Accept ``[x0, y0, x1, y1]`` or ``((x0, y0), (x1, y1))``."""
    if not isinstance(value, (list, tuple)):
        return None
    if len(value) == 2 and all(isinstance(item, (list, tuple)) and len(item) == 2 for item in value):
        (x0, y0), (x1, y1) = value
        candidate = (x0, y0, x1, y1)
    elif len(value) == 4 and all(isinstance(item, (int, float)) for item in value):
        candidate = tuple(value)
    else:
        return None
    try:
        x0, y0, x1, y1 = (float(item) for item in candidate)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in (x0, y0, x1, y1)):
        return None
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _iter_word_dicts(payload: Any) -> Iterable[dict]:
    if isinstance(payload, dict):
        for key in ("words", "items", "entries"):
            if isinstance(payload.get(key), (list, tuple)):
                yield from _iter_word_dicts(payload[key])
                return
        if any(key in payload for key in _BOX_KEYS):
            yield payload
        return
    if isinstance(payload, (list, tuple)):
        for item in payload:
            yield from _iter_word_dicts(item)


def parse_page_words(
    page_words: Any, page_index: int, page_size: tuple[float, float] | None
) -> list[Word]:
    """Extract the words of one page, rescaling unit coordinates when needed."""
    if not isinstance(page_words, (list, tuple)) or page_index >= len(page_words):
        return []
    raw = list(_iter_word_dicts(page_words[page_index]))
    if not raw:
        return []

    parsed: list[tuple[str, tuple[float, float, float, float], float | None]] = []
    for entry in raw:
        box = None
        for key in _BOX_KEYS:
            if key in entry:
                box = _coerce_box(entry[key])
                if box is not None:
                    break
        if box is None:
            continue
        text = ""
        for key in _TEXT_KEYS:
            candidate = entry.get(key)
            if isinstance(candidate, str):
                text = candidate.strip()
                break
        try:
            confidence = float(entry["confidence"])
        except (KeyError, TypeError, ValueError):
            confidence = None
        if confidence is not None and not math.isfinite(confidence):
            confidence = None
        parsed.append((text, box, confidence))

    if not parsed:
        return []

    # Unit-normalised coordinates are rescaled to the measured page; without a
    # measured page they cannot be placed, so the page yields no geometry.
    largest = max(max(box[2], box[3]) for _text, box, _c in parsed)
    if largest <= 1.5:
        if page_size is None:
            return []
        width, height = page_size
        parsed = [
            (text, (box[0] * width, box[1] * height, box[2] * width, box[3] * height), conf)
            for text, box, conf in parsed
        ]

    return [Word(text, *box, conf) for text, box, conf in parsed]


def group_lines(words: list[Word]) -> list[list[Word]]:
    """Group words into text lines by vertical proximity of their centres."""
    if not words:
        return []
    heights = [word.height for word in words if word.height > 0]
    tolerance = (statistics.median(heights) * 0.6) if heights else 1.0
    lines: list[list[Word]] = []
    for word in sorted(words, key=lambda item: (item.y_center, item.x0)):
        if lines:
            current = lines[-1]
            reference = statistics.median([item.y_center for item in current])
            if abs(word.y_center - reference) <= tolerance:
                current.append(word)
                continue
        lines.append([word])
    return [sorted(line, key=lambda item: item.x0) for line in lines]


def _cluster(values: list[float], tolerance: float) -> list[list[float]]:
    clusters: list[list[float]] = []
    for value in sorted(values):
        if clusters and value - clusters[-1][-1] <= tolerance:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return clusters


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not denominator:
        return 0.0
    result = numerator / denominator
    return result if math.isfinite(result) else 0.0


#: A newspaper column is narrower than a page and wider than a caption.
#: Anything spanning most of the width is a headline, a masthead or a rule, and
#: is excluded from the column estimate rather than merging every column into
#: one band.
_COLUMN_SPAN_LIMIT = 0.75
_COLUMN_MIN_LINES = 4
_COLUMN_MIN_WORDS_PER_LINE = 4


def estimate_narrative_columns(
    lines: list[list[Word]], page_width: float, page_height: float
) -> int:
    """Count narrative text columns on one page, from word positions alone.

    ``two_column_ratio`` answers a yes/no question about two columns and says
    nothing about three, four or five, which is what a newspaper front page
    actually has. This counts them: cluster the left edge of every *body* line
    and count the clusters that carry a column's worth of lines.

    Lines that span most of the page are excluded — a banner headline crosses
    every column and would otherwise merge them — and so are the top and bottom
    bands, where a running header or a folio sits alone on the line.
    """
    if page_width <= 0 or page_height <= 0 or not lines:
        return 0
    tolerance = max(4.0, 0.015 * page_width)
    starts: list[float] = []
    for line in lines:
        if len(line) < _COLUMN_MIN_WORDS_PER_LINE:
            continue
        start = line[0].x0
        end = max(word.x1 for word in line)
        if (end - start) >= _COLUMN_SPAN_LIMIT * page_width:
            continue
        centre = statistics.median([word.y_center for word in line])
        if centre <= 0.06 * page_height or centre >= 0.96 * page_height:
            continue
        starts.append(start)

    if len(starts) < 2 * _COLUMN_MIN_LINES:
        return 0
    return sum(
        1 for cluster in _cluster(starts, tolerance) if len(cluster) >= _COLUMN_MIN_LINES
    )


def _page_geometry(words: list[Word], page_size: tuple[float, float]) -> dict | None:
    page_width, page_height = page_size
    lines = group_lines(words)
    if not lines or not math.isfinite(page_width) or not math.isfinite(page_height):
        return None
    if page_width <= 0 or page_height <= 0:
        return None

    column_tolerance = max(3.0, 0.012 * page_width)
    heights = [word.height for word in words if word.height > 0]
    median_height = statistics.median(heights) if heights else 10.0

    # --- estimating the width of an ordinary space ------------------------
    # Not the median gap: on a form most gaps are the wide ones separating a
    # label from its value, so the median *is* the wide gap and nothing ever
    # exceeds a multiple of it. A low quantile, floored by a fraction of the
    # glyph height, estimates the ordinary inter-word space on both prose and
    # sparse layouts.
    gaps: list[float] = []
    for line in lines:
        for previous, following in zip(line, line[1:]):
            gaps.append(max(0.0, following.x0 - previous.x1))
    positive_gaps = sorted(gap for gap in gaps if gap > 0)
    if positive_gaps:
        quantile = positive_gaps[min(len(positive_gaps) - 1, int(0.20 * len(positive_gaps)))]
        space_width = max(quantile, 0.20 * median_height)
    else:
        space_width = 0.25 * median_height
    wide_gap_floor = max(3.0 * space_width, 0.04 * page_width)

    # --- segment starts, tab stops and label-value pairs ------------------
    # A "segment" starts at the beginning of a line, after a wide gap, or at the
    # value after a ``Label:`` token. In running prose every line contributes
    # exactly one segment, so there is one column: the left margin. In a field
    # layout each line contributes a segment per column, and those columns repeat
    # down the page. Clustering *segment* starts rather than every word start is
    # what keeps prose from registering as a grid — the previous version counted
    # every word and reported eight columns for a paragraph.
    segment_starts_by_line: list[list[float]] = []
    wide_gap_flags: list[bool] = []
    wide_gap_lines = 0
    label_value_lines = 0
    for line in lines:
        starts = [line[0].x0]
        has_wide_gap = False
        for previous, following in zip(line, line[1:]):
            if following.x0 - previous.x1 >= wide_gap_floor:
                starts.append(following.x0)
                has_wide_gap = True
        segment_starts_by_line.append(starts)
        wide_gap_flags.append(has_wide_gap)
        if has_wide_gap:
            wide_gap_lines += 1
        # A label-value pair: a colon-terminated token followed by a clear gap,
        # or trailing a line with nothing after it (an unfilled field).
        for index, word in enumerate(line):
            if not word.text.endswith(":"):
                continue
            if index == len(line) - 1:
                label_value_lines += 1
                break
            following = line[index + 1]
            label_value_floor = max(space_width, 0.015 * page_width)
            if following.x0 - word.x1 >= label_value_floor:
                label_value_lines += 1
                if following.x0 - word.x1 < wide_gap_floor:
                    starts.append(following.x0)
                break

    all_starts = [start for starts in segment_starts_by_line for start in starts]
    columns = []
    for cluster in _cluster(all_starts, column_tolerance):
        centre = statistics.median(cluster)
        lines_touching = sum(
            1
            for starts in segment_starts_by_line
            if any(abs(start - centre) <= column_tolerance for start in starts)
        )
        if lines_touching >= min(3, len(lines)):
            columns.append(centre)
    alignment_columns = len(columns)
    tab_stops = max(0, alignment_columns - 1)  # the leftmost column is the margin

    # --- line shape -------------------------------------------------------
    body_lines = [line for line in lines if len(line) >= 5]
    right_edges = [max(word.x1 for word in line) for line in body_lines]
    if len(right_edges) >= 3:
        right_edge_regularity = 1.0 - min(1.0, statistics.pstdev(right_edges) / (0.25 * page_width))
    else:
        right_edge_regularity = 0.0

    centres = [statistics.median([word.y_center for word in line]) for line in lines]
    pitches = [second - first for first, second in zip(centres, centres[1:]) if second > first]
    if len(pitches) >= 3:
        # Median absolute deviation, not standard deviation: one large jump —
        # the gap between a title and the body, or a section break — is normal
        # in an otherwise perfectly regular page, and under a squared measure
        # that single outlier drove the metric to zero for every document that
        # had a heading.
        median_pitch = statistics.median(pitches)
        deviation = statistics.median([abs(pitch - median_pitch) for pitch in pitches])
        variation = deviation / median_pitch if median_pitch else 1.0
        line_pitch_regularity = 1.0 - min(1.0, variation)
    else:
        line_pitch_regularity = 0.0

    line_starts = [line[0].x0 for line in lines]
    left_margin = statistics.median(line_starts)
    indented = sum(1 for start in line_starts if start > left_margin + 0.02 * page_width)

    short_lines = sum(
        1 for line in lines if (max(w.x1 for w in line) - line[0].x0) < 0.60 * page_width
    )

    # A narrative line is a *run of prose*: enough words on one line, and not
    # broken into columns by a wide gap. Forms, slides, tables of entries and
    # captioned visuals all produce lines that are short, gapped, or both. The
    # ratio is what separates "little text" (which says nothing about a family)
    # from "little narrative text" (which does), so no rule needs to reach for
    # raw word counts or OCR quality as a stand-in for layout.
    narrative_lines = sum(
        1
        for line, has_gap in zip(lines, wide_gap_flags)
        if len(line) >= 8 and not has_gap
    )

    page_centre = page_width / 2.0
    centered = 0
    for line in lines:
        start, end = line[0].x0, max(word.x1 for word in line)
        width = end - start
        if width < 0.70 * page_width and abs((start + end) / 2.0 - page_centre) <= 0.08 * page_width:
            centered += 1

    # A header block is short lines at the top of the page *against* a page
    # whose body is not short. Measured as an excess over the document's own
    # short-line rate, so a uniformly sparse page does not read as a header.
    top_band = [line for line, centre in zip(lines, centres) if centre <= 0.20 * page_height]
    top_band_short = sum(
        1 for line in top_band if (max(w.x1 for w in line) - line[0].x0) < 0.60 * page_width
    )
    body_index = [
        index for index, centre in enumerate(centres) if centre > 0.20 * page_height
    ]
    body_band = [lines[index] for index in body_index]
    body_short = sum(
        1 for line in body_band if (max(w.x1 for w in line) - line[0].x0) < 0.60 * page_width
    )
    # Wide gaps below the header band. A letter's header block is *made* of wide
    # gaps, so measuring them over the whole page cannot distinguish a letter
    # from a form; measuring them over the body can — a letter's body is
    # uninterrupted prose, a form's body is columns all the way down.
    body_wide_gap = sum(1 for index in body_index if wide_gap_flags[index])
    header_excess = max(
        0.0,
        _safe_ratio(top_band_short, len(top_band)) - _safe_ratio(body_short, len(body_band)),
    )

    narrative_column_count = estimate_narrative_columns(lines, page_width, page_height)

    words_flat = [word for line in lines for word in line]
    uppercase = sum(
        1 for word in words_flat if len(word.text) >= 3 and word.text.isupper()
    )
    confidences = [word.confidence for word in words_flat if word.confidence is not None]

    return {
        "lines": len(lines),
        "words": len(words_flat),
        "alignment_columns": alignment_columns,
        "tab_stops": tab_stops,
        "wide_gap_line_ratio": _safe_ratio(wide_gap_lines, len(lines)),
        "body_wide_gap_line_ratio": _safe_ratio(body_wide_gap, len(body_band)) if body_band else 0.0,
        "label_value_line_ratio": _safe_ratio(label_value_lines, len(lines)),
        "label_value_lines": label_value_lines,
        "right_edge_regularity": round(max(0.0, right_edge_regularity), 4),
        "line_pitch_regularity": round(max(0.0, line_pitch_regularity), 4),
        "indent_ratio": _safe_ratio(indented, len(lines)),
        "short_line_ratio": _safe_ratio(short_lines, len(lines)),
        "narrative_line_ratio": _safe_ratio(narrative_lines, len(lines)),
        "narrative_column_count": narrative_column_count,
        "centered_line_ratio": _safe_ratio(centered, len(lines)),
        "top_band_header_ratio": round(header_excess, 4),
        "space_width": round(space_width, 2),
        "uppercase_word_ratio": _safe_ratio(uppercase, len(words_flat)),
        "word_confidence_mean": (
            round(statistics.fmean(confidences), 4) if confidences else None
        ),
    }


class PageLine(NamedTuple):
    """One reconstructed OCR line, with everything a lexical rule needs.

    Case and punctuation are preserved verbatim — a byline is recognised by
    ``BY`` in capitals and a dateline by its comma, so a lower-cased or
    stripped reconstruction would destroy the very signal it is built to find.
    """

    text: str
    page_index: int
    x0: float
    y0: float
    x1: float
    y1: float
    page_width: float
    page_height: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0


def _page_size_of(page_sizes: Any, page_index: int) -> tuple[float, float] | None:
    sizes = page_sizes if isinstance(page_sizes, (list, tuple)) else []
    if page_index >= len(sizes):
        return None
    candidate = sizes[page_index]
    if not isinstance(candidate, (list, tuple)) or len(candidate) != 2:
        return None
    try:
        width, height = float(candidate[0]), float(candidate[1])
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(width) and math.isfinite(height)) or width <= 0 or height <= 0:
        return None
    return width, height


def reconstruct_page_lines(
    page_words: Any, page_sizes: Any, page_count: int
) -> list[PageLine]:
    """Rebuild readable lines of text from the word boxes, page by page.

    ``page_words`` was feeding the geometry features only, so everything the
    OCR read but the layout detector did not enclose in a region was invisible
    to every lexical rule. On a scanned broadsheet that is most of the editorial
    furniture: the nameplate, the dateline, the edition line and the bylines sit
    in bands the region detector types as something else or misses entirely.

    This is the *same* parser the geometry features use — the words are read by
    :func:`parse_page_words` and grouped by :func:`group_lines` — so the two
    views of the page cannot drift apart, which a second, independent word
    reader would guarantee they eventually did.
    """
    try:
        total_pages = max(0, int(page_count))
    except (TypeError, ValueError, OverflowError):
        return []

    lines: list[PageLine] = []
    for page_index in range(total_pages):
        size = _page_size_of(page_sizes, page_index)
        words = parse_page_words(page_words, page_index, size)
        if not words:
            continue
        if size is None:
            size = (
                max(word.x1 for word in words) * 1.06,
                max(word.y1 for word in words) * 1.06,
            )
        width, height = size
        for line in group_lines(words):
            text = " ".join(word.text for word in line if word.text).strip()
            if not text:
                continue
            lines.append(
                PageLine(
                    text=text,
                    page_index=page_index,
                    x0=min(word.x0 for word in line),
                    y0=min(word.y0 for word in line),
                    x1=max(word.x1 for word in line),
                    y1=max(word.y1 for word in line),
                    page_width=width,
                    page_height=height,
                )
            )
    return lines


#: A nameplate is set several times the size of body type, spans a good part of
#: the measure, and sits at the top of the page. These are the three properties
#: that hold for every newspaper nameplate and for almost nothing else.
MASTHEAD_MIN_HEIGHT_RATIO = 2.2
MASTHEAD_MIN_WIDTH_FRACTION = 0.25
MASTHEAD_TOP_BAND = 0.25


def find_masthead_candidates(lines: list[PageLine]) -> list[PageLine]:
    """Typographic nameplate candidates: size and position, never wording.

    Deliberately blind to what the line *says*. A blackletter nameplate read as
    "Che New Hork Cimes" is the same typographic object as a correct reading,
    and a detector that required the name to be recognised would fail on
    exactly the mastheads that are hardest to read — which are the ones set in
    display faces, which is to say the ones most likely to be mastheads.

    It is equally blind to the region class: the layout detector is free to call
    the nameplate a ``Picture``, a ``Text`` block or nothing at all.
    """
    if not lines:
        return []
    body_heights = [line.height for line in lines if line.height > 0]
    if not body_heights:
        return []
    median_height = statistics.median(body_heights)
    if median_height <= 0:
        return []

    candidates = []
    for line in lines:
        if line.page_height <= 0 or line.page_width <= 0:
            continue
        if line.y0 > MASTHEAD_TOP_BAND * line.page_height:
            continue
        if line.height < MASTHEAD_MIN_HEIGHT_RATIO * median_height:
            continue
        if line.width < MASTHEAD_MIN_WIDTH_FRACTION * line.page_width:
            continue
        candidates.append(line)
    return candidates


def extract_geometry_features(
    page_words: Any,
    page_sizes: Any,
    page_count: int,
) -> dict:
    """Aggregate word-level geometry across the document.

    Ratios are averaged over the pages that yielded geometry, weighted by line
    count so a dense page is not outvoted by a nearly empty one. Counts that
    describe structure rather than volume (alignment columns, tab stops) are
    reported as the maximum over pages: one clearly gridded page is evidence of
    a gridded document, and averaging would dilute it away.

    ``geometry_page_ratio`` reports how much of the document this describes.
    When it is zero the whole channel is unavailable and no geometric rule
    should be scored — nor counted against a family in the denominator.
    """
    try:
        total_pages = max(0, int(page_count))
    except (TypeError, ValueError, OverflowError):
        total_pages = 0

    per_page: list[dict] = []
    sizes = page_sizes if isinstance(page_sizes, (list, tuple)) else []
    for page_index in range(total_pages):
        size = None
        if page_index < len(sizes):
            candidate = sizes[page_index]
            if isinstance(candidate, (list, tuple)) and len(candidate) == 2:
                try:
                    width, height = float(candidate[0]), float(candidate[1])
                except (TypeError, ValueError):
                    width = height = 0.0
                if math.isfinite(width) and math.isfinite(height) and width > 0 and height > 0:
                    size = (width, height)
        words = parse_page_words(page_words, page_index, size)
        if not words:
            continue
        if size is None:
            # Fall back to the words' own extent only for the ratios that are
            # scale-free; a page whose size is unknown still has a usable
            # internal geometry.
            size = (
                max(word.x1 for word in words) * 1.06,
                max(word.y1 for word in words) * 1.06,
            )
        geometry = _page_geometry(words, size)
        if geometry:
            per_page.append(geometry)

    if not per_page:
        return dict(_DEFAULT_GEOMETRY_FEATURES)

    total_lines = sum(page["lines"] for page in per_page) or 1

    def weighted(key: str) -> float:
        return round(
            sum(page[key] * page["lines"] for page in per_page) / total_lines, 4
        )

    confidences = [
        page["word_confidence_mean"] for page in per_page if page["word_confidence_mean"] is not None
    ]
    return {
        "geometry_pages": len(per_page),
        "geometry_page_ratio": round(len(per_page) / max(1, total_pages), 4),
        "word_line_count": sum(page["lines"] for page in per_page),
        "word_token_count": sum(page["words"] for page in per_page),
        "alignment_column_count": max(page["alignment_columns"] for page in per_page),
        "tab_stop_count": max(page["tab_stops"] for page in per_page),
        "label_value_line_count": sum(page["label_value_lines"] for page in per_page),
        "label_value_line_ratio": weighted("label_value_line_ratio"),
        "wide_gap_line_ratio": weighted("wide_gap_line_ratio"),
        "body_wide_gap_line_ratio": weighted("body_wide_gap_line_ratio"),
        "right_edge_regularity": weighted("right_edge_regularity"),
        "line_pitch_regularity": weighted("line_pitch_regularity"),
        "indent_ratio": weighted("indent_ratio"),
        "short_line_ratio": weighted("short_line_ratio"),
        "narrative_line_ratio": weighted("narrative_line_ratio"),
        # Column structure is a per-page fact, so it is reported as a maximum
        # and a ratio rather than averaged: one three-column page is evidence of
        # a three-column publication, and averaging it against a full-page
        # advertisement would dilute it away.
        "narrative_column_count": max(page["narrative_column_count"] for page in per_page),
        "newspaper_column_pages": sum(
            1 for page in per_page if page["narrative_column_count"] >= 2
        ),
        "newspaper_column_geometry_ratio": round(
            sum(1 for page in per_page if page["narrative_column_count"] >= 2)
            / len(per_page),
            4,
        ),
        "centered_line_ratio": weighted("centered_line_ratio"),
        "top_band_header_ratio": weighted("top_band_header_ratio"),
        "space_width": weighted("space_width"),
        "uppercase_word_ratio": weighted("uppercase_word_ratio"),
        "word_confidence_mean": (
            round(statistics.fmean(confidences), 4) if confidences else None
        ),
    }