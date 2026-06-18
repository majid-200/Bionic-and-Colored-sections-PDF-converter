"""
Bionic Reader for Research Papers
===================================
Converts a PDF into a Bionic Reading format:
  - Bolds the first ~45% of each word (configurable)
  - Colors each sentence alternately so you can visually track start → end
  - Preserves images, equations, and figures exactly (drawn as page background)
  - Handles single- and multi-column layouts via PyMuPDF's layout engine

Requirements:
    pymupdf

Usage (Windows):
    python bionic_reader.py paper.pdf
    python bionic_reader.py paper.pdf output.pdf --bold-ratio 0.5
    python bionic_reader.py paper.pdf --pages 1-10 --sentence-colors "#0A3D8F,#1A5C2E"

Options:
    input                  Source PDF path
    output                 Output PDF path  [default: <input>_bionic.pdf]
    --bold-ratio  FLOAT    Fraction of each word to bold  [default: 0.45]
    --sentence-colors HEXES Comma-sep hex colors, cycled per sentence
                           [default: deep-navy / forest-green]
    --pages  RANGE         Pages to process e.g. "1-5" or "2,4,6"  [default: all]
    --dpi  INT             Resolution for rendering background image  [default: 150]
    --no-skip-math         Apply bionic to math spans too (off by default)
"""

from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path
from typing import NamedTuple

try:
    import pymupdf as fitz  # requires pymupdf >= 1.24
except ImportError:
    sys.exit("PyMuPDF not found.  Run:  pip install pymupdf")


# ── Defaults ──────────────────────────────────────────────────────────────────

# Default sentence alternation colours as (R, G, B) float tuples in [0.0, 1.0].
# Sentences will cycle through these colours so readers can easily track where
# each sentence starts and ends.
DEFAULT_COLORS = [
    (0.05, 0.20, 0.55),   # deep navy
    (0.08, 0.42, 0.18),   # forest green
    (0.85, 0.65, 0.15),   # mustard gold
    (0.80, 0.35, 0.20),   # burnt terracotta
    (0.73, 0.43, 0.76),   # lavender purple
    (0.15, 0.20, 0.25),   # charcoal gray
    (0.04, 0.75, 0.52),   # teal mint
    (0.78, 0.42, 0.00),   # copper bronze
]

# Substrings found in font names that indicate a mathematical/symbol font.
# When a span uses one of these fonts we treat the whole span as math and
# skip bionic formatting to avoid mangling equations.
MATH_FONT_FRAGMENTS = {
    "symbol", "zapf", "mt extra", "cambria math", "stix",
    "asana", "tex", "mathjax", "cmr", "cmsy", "cmex", "cmmi",
    "msam", "msbm", "eulervm", "fourier",
}

# Regex that matches Unicode characters associated with mathematics:
#   - Miscellaneous math operators (∀–⋿)
#   - Arrows, technical symbols (⌀–⏿)
#   - Enclosed alphanumerics (①–⑳)
#   - Greek uppercase/lowercase letters
# Any span whose text contains these characters is flagged as math.
MATH_CHAR_RE = re.compile(
    r"[∀-⋿⌀-⏿①-⑳←-⇿∀-∿⊀-⊿⋀-⋿"
    r"αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ]"
)

# Common abbreviations whose trailing period must NOT be treated as a sentence
# boundary. Without this list "Fig. 3" or "et al." would incorrectly split
# into two sentences.
ABBREVS = frozenset({
    "fig", "figs", "eq", "eqs", "et", "al", "i.e", "e.g", 
    "dr", "mr", "mrs", "ms", "prof", "vs", "approx",
    "sec", "ref", "tab", "no", "vol", "pp", "cf", "proc",
    "dept", "univ", "est", "prob", "avg", "max", "min",
})


# ── Data classes ──────────────────────────────────────────────────────────────

class Word(NamedTuple):
    """Immutable record for a single typographic word extracted from a PDF span."""
    text: str          # the raw word string (may include punctuation)
    origin: tuple      # (x, y) PDF-space baseline of the first character
    bbox: fitz.Rect    # tight bounding box enclosing all characters in the word
    size: float        # font size in points
    font: str          # font name exactly as stored in the PDF
    is_math: bool      # True when heuristics flag this as a math/symbol token


# ── Helper utilities ──────────────────────────────────────────────────────────

def hex_to_rgb(h: str) -> tuple[float, float, float]:
    """
    Convert a CSS hex colour string to a (R, G, B) tuple with components in [0, 1].
    Accepts both 3-digit shorthand (#RGB) and 6-digit (#RRGGBB) forms.
    """
    h = h.lstrip("#")
    if len(h) == 3:
        # Expand shorthand: "#ABC" → "AABBCC"
        h = h[0]*2 + h[1]*2 + h[2]*2
    # Slice into three 2-hex pairs and normalise to [0, 1]
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def parse_pages(spec: str, n: int) -> list[int]:
    """
    Parse a page-range specification string into a sorted list of 0-indexed
    page numbers that are valid within a document of `n` pages.

    Supported formats:
        "1-5"    → pages 1 through 5 inclusive  → [0, 1, 2, 3, 4]
        "2,4,6"  → individual pages              → [1, 3, 5]
        "1-3,7"  → mixed                         → [0, 1, 2, 6]

    The input uses 1-based page numbers (as displayed in a PDF viewer);
    this function converts them to 0-based indices for PyMuPDF.
    """
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a) - 1, int(b)))   # convert 1-based → 0-based
        else:
            out.add(int(part) - 1)
    # Filter out any indices that fall outside the actual document
    return sorted(p for p in out if 0 <= p < n)


def n_bold_chars(word: str, ratio: float) -> int:
    """
    Return how many leading characters of `word` should be rendered bold.

    The ratio applies only to *alphabetic* characters (ignoring digits and
    punctuation), which matches the original Bionic Reading specification.
    The function then maps the target alphabetic count back to the raw string
    index so the caller can simply slice word[:n] / word[n:].

    Examples (ratio=0.45):
        "the"   → 1 bold char  ("t")
        "brain" → 2 bold chars ("br")
        "won't" → 2 bold chars ("wo") — apostrophe doesn't count
    """
    alpha = [c for c in word if c.isalpha()]
    # Single-character words are always fully bolded
    if len(alpha) <= 1:
        return len(word)
    # Round to nearest integer, minimum 1
    target = max(1, round(len(alpha) * ratio))
    # Walk the raw word string, counting only alphabetic characters
    count = 0
    for i, ch in enumerate(word):
        if ch.isalpha():
            count += 1
            if count >= target:
                return i + 1   # return the raw-string index after this char
    return len(word)


def is_math(span_font: str, span_text: str) -> bool:
    """
    Heuristic test for whether a text span is likely mathematical content.

    Returns True if any of the following hold:
    1. The font name contains a known math-font substring.
    2. The span text contains Unicode characters from math/Greek ranges.
    3. The span is ≤ 4 characters composed entirely of digits and math operators
       (catches inline numerals like "2", "+1", "≈3").
    """
    fl = span_font.lower()
    # Check 1: math font name
    if any(f in fl for f in MATH_FONT_FRAGMENTS):
        return True
    # Check 2: Unicode math / Greek characters in the text
    if MATH_CHAR_RE.search(span_text):
        return True
    # Check 3: short digit/operator-only tokens
    stripped = span_text.strip()
    if len(stripped) <= 4 and re.fullmatch(r'[\d\+\-\*/=<>(){}|^_.,;\\ ]+', stripped):
        return True
    return False


def get_font(name: str, bold: bool = False) -> fitz.Font:
    """
    Resolve a fitz.Font by name, with graceful fallback to Helvetica variants.

    PDF font names often carry a 6-character subset prefix (e.g.
    "ABCDEF+TimesNewRoman"). This function strips that prefix before attempting
    resolution, then tries the cleaned name, then the original, then the
    Helvetica fallback.

    Args:
        name: Font name as stored in the PDF span dict.
        bold: If True, use a bold variant for fallback.

    Returns:
        A fitz.Font instance — never raises; always returns something usable.
    """
    # fallback = "hebo" if bold else "helv"   # Helvetica-Bold / Helvetica
    fallback = "tibo" if bold else "tiro"   # Times-Bold / Times-Roman
    # Strip the 6-char subset prefix that PDF embedders add
    clean = name.split("+")[-1]
    for candidate in (clean, name, fallback):
        try:
            return fitz.Font(fontname=candidate)
        except Exception:
            pass
    # Last resort — this should never be reached given "helv"/"hebo" always work
    return fitz.Font(fontname=fallback)


# ── Word extraction ───────────────────────────────────────────────────────────

def extract_words(page: fitz.Page, skip_math: bool) -> list[Word]:
    """
    Extract every typographic word from `page` with character-level precision.

    Uses PyMuPDF's "rawdict" mode which exposes individual character bounding
    boxes, enabling sub-word layout accuracy for the bionic overlay.

    The document structure traversed is:
        page → blocks (text or image) → lines → spans → chars

    Each span (a run of characters sharing the same font/size/colour) is split
    at whitespace boundaries to produce individual Word records.

    Args:
        page:      The PyMuPDF page to extract from.
        skip_math: If True, spans identified as mathematical are flagged so the
                   caller can skip bionic formatting for them.

    Returns:
        Flat list of Word namedtuples in reading order.
    """
    words: list[Word] = []
    # TEXT_PRESERVE_WHITESPACE keeps space chars so we can detect word boundaries
    blocks = page.get_text("rawdict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]

    for block in blocks:
        # block["type"] == 0 is text; 1 is an embedded image — skip images
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            if line["dir"][1] != 0:
                print(line)
                print(line["dir"])
                print('------------------------------------')
            for span in line["spans"]:
                font_name = span["font"]
                size      = span["size"]
                chars     = span["chars"]   # list of {"c", "origin", "bbox"}

                # Reconstruct full span text for the math heuristic check
                span_text = "".join(ch["c"] for ch in chars)
                # Only flag as math when skip_math is enabled
                math_flag = skip_math and is_math(font_name, span_text)

                # Accumulate characters into words, flushing on whitespace
                current_chars: list[dict] = []
                for ch in chars:
                    if ch["c"] in (" ", "\t", "\n", "\r"):
                        # Whitespace signals the end of the current word
                        if current_chars:
                            _flush_word(current_chars, font_name, size,
                                        math_flag, words)
                            current_chars = []
                    else:
                        current_chars.append(ch)
                # Flush any trailing word that wasn't followed by whitespace
                if current_chars:
                    _flush_word(current_chars, font_name, size, math_flag, words)

    return words


def _flush_word(chars: list[dict], font: str, size: float,
                math_flag: bool, out: list[Word]) -> None:
    """
    Finalise a collected group of characters into a Word and append it to `out`.

    Computes the tight bounding box as the union of all per-character bboxes,
    and uses the first character's origin as the word's typographic baseline.

    Args:
        chars:     Non-whitespace character dicts from a single span.
        font:      Font name of the containing span.
        size:      Font size in points of the containing span.
        math_flag: Whether this word is flagged as mathematical content.
        out:       Output list to append the resulting Word to.
    """
    text = "".join(ch["c"] for ch in chars)
    if not text.strip():
        return   # skip degenerate whitespace-only entries
    # Build the union bounding box from individual character rects
    x0 = min(ch["bbox"][0] for ch in chars)
    y0 = min(ch["bbox"][1] for ch in chars)
    x1 = max(ch["bbox"][2] for ch in chars)
    y1 = max(ch["bbox"][3] for ch in chars)
    bbox = fitz.Rect(x0, y0, x1, y1)
    # The typographic origin is the baseline of the first character
    origin = (chars[0]["origin"][0], chars[0]["origin"][1])
    out.append(Word(text=text, origin=origin, bbox=bbox,
                    size=size, font=font, is_math=math_flag))


# ── Sentence splitting ─────────────────────────────────────────────────────────

def split_sentences(words: list[Word]) -> list[list[Word]]:
    """
    Group a flat list of Words into sentences based on terminal punctuation.

    Strategy:
    - A word ending in `.`, `!`, or `?` (optionally followed by a closing quote
      or bracket) terminates the current sentence.
    - Words whose stripped, de-punctuated core appears in ABBREVS (e.g. "fig",
      "et") are NOT treated as sentence boundaries.
    - Single lowercase letters followed by a period (e.g. section labels like
      "a.") are also not treated as sentence boundaries.

    This heuristic works well for scientific papers but is not a full NLP
    sentence tokenizer — edge cases like "Dr. Smith arrived." may occasionally
    split incorrectly.

    Returns:
        A list of sentence groups, where each group is a list of Words.
        Words that appear after the last terminal punctuation form a final
        "open" sentence that is still included.
    """
    sentences: list[list[Word]] = []
    current: list[Word] = []
    for w in words:
        current.append(w)
        t = w.text
        # Match terminal punctuation, optionally followed by closing delimiters
        if re.search(r'[.!?]["\')\]]?$', t):
            # Strip the terminal punctuation to get the core word for abbrev check
            core = t.rstrip('.!?"\'()[]').lower()
            # Only split if it's NOT an abbreviation and NOT a single lowercase letter
            if core not in ABBREVS and not re.fullmatch(r'[a-z]', core):
                sentences.append(current)
                current = []
    # Any remaining words after the last sentence boundary form an open sentence
    if current:
        sentences.append(current)
    return sentences


# ── Page processing ────────────────────────────────────────────────────────────

def process_page(page: fitz.Page, bold_ratio: float, colors: list,
                 sentence_idx: list[int], skip_math: bool) -> None:
    """
    Transform a single page into Bionic Reading format in-place.

    The transformation is a five-phase overlay pipeline:

    Phase 1 — Rasterise:
        Render the original page to a pixmap. This preserves all figures,
        equations, and graphical elements that we cannot reconstruct as vectors.

    Phase 2 — Extract:
        Pull word positions from the original page content *before* modifying it.
        We need the original text positions to know where to write the new text.

    Phase 3 — Blank + background:
        Erase all page content streams and reinsert the rasterised pixmap as a
        full-page background image. The page is now a flat image.

    Phase 4 — White knockouts:
        Draw semi-transparent white rectangles over each word's bounding box.
        This erases the blurry rasterised text pixels so the new crisp vector
        text written in phase 5 is clearly readable.

    Phase 5 — Bionic text:
        Use fitz.TextWriter to place each word's bold prefix and normal suffix
        at the exact PDF-space coordinates extracted in phase 2, coloured by
        sentence index.

    Args:
        page:         The fitz.Page to modify in-place.
        bold_ratio:   Fraction [0, 1] of alphabetic characters to bold.
        colors:       List of (R, G, B) tuples to cycle through per sentence.
        sentence_idx: Mutable single-element list used as a cross-page counter.
                      Passed by reference so the colour sequence is continuous
                      across all processed pages.
        skip_math:    If True, math-flagged words are skipped in phases 4 and 5,
                      leaving their rasterised appearance from the background.
    """

    # ── Phase 1: Rasterise the original page ─────────────────────────────────
    # Scale 1.5× gives ~108 DPI, balancing quality against file size.
    # alpha=False produces an RGB pixmap (no transparency needed for background).
    # mat = fitz.Matrix(2.0, 2.0)
    # pix = page.get_pixmap(matrix=mat, alpha=False)

    # ── Phase 2: Extract text positions from the untouched page ───────────────
    # Must happen BEFORE clean_contents() destroys the original content streams.
    words = extract_words(page, skip_math)
    sentences = split_sentences(words)

    # ── Phase 3: Blank the page and reinsert rasterised background ────────────
    # clean_contents() removes all existing PDF operators (text, paths, images).
    # insert_image() places the pixmap to fill the entire page rectangle,
    # effectively converting the page to a flat image base layer.
    # page.clean_contents()
    rect = page.rect
    # page.insert_image(rect, pixmap=pix, keep_proportion=False)

    # ── Phase 4: White knockout rectangles behind each text word ──────────────
    # Without this, the bionic vector text would sit on top of blurry rasterised
    # text from the background image, making it hard to read.
    # fill_opacity=0.88 retains a slight hint of the background (useful when
    # coloured or shaded text boxes exist in the original).
    shape = page.new_shape()
    for sent in sentences:
        for w in sent:
            if w.is_math:
                continue   # leave math regions as rasterised — do not blank them
            # Expand the word bbox by 1.2 pt on each side to avoid clipping
            # descenders (e.g. 'g', 'p', 'y') and tall ascenders ('h', 'l')
            pad = 0
            r = fitz.Rect(w.bbox.x0 - pad, w.bbox.y0 - pad,
                           w.bbox.x1 + pad, w.bbox.y1 + pad)
            shape.draw_rect(r)
            shape.finish(fill=(1, 1, 1), fill_opacity=1,
                         color=None, width=0)
    shape.commit()   # flush all drawn paths to the page in a single operation

    # ── Phase 5: Write bionic-formatted text ──────────────────────────────────
    # TextWriter batches all glyph operations and flushes them in one call to
    # write_text(), which is more efficient than inserting each word separately.
    # writer = fitz.TextWriter(rect)
    # 1. Setup a dedicated TextWriter instance for EVERY color in your list
    writers = {col: fitz.TextWriter(rect) for col in colors}


    for sent in sentences:
        # Pick the next colour in the cycle. sentence_idx[0] is the running
        # global counter; modulo wraps it back when we exceed len(colors).
        col = colors[sentence_idx[0] % len(colors)]
        sentence_idx[0] += 1   # advance counter so each sentence gets a new colour

        # Select the specific writer mapped to this color
        active_writer = writers[col]

        for w in sent:
            # print(w)
            if w.is_math:
                continue   # math tokens remain as background image pixels
            text = w.text
            if not text.strip():
                continue

            # Split the word into bold prefix and normal-weight suffix
            nb     = n_bold_chars(text, bold_ratio)
            prefix = text[:nb]    # characters to render bold
            suffix  = text[nb:]   # remaining characters at normal weight

            # Get the appropriate font objects for this word's original font
            bold_font   = get_font(w.font, bold=True)
            normal_font = get_font(w.font, bold=False)
        
            # ox/oy is the typographic baseline origin for this word.
            # After appending the bold prefix, ox is advanced by the rendered
            # width of the prefix so the suffix continues seamlessly.
            ox, oy = w.origin
            font_offset = 0.5
            try:
                if prefix:
                    active_writer.append((ox, oy), prefix,
                                         font=bold_font, fontsize=w.size - font_offset)
                    # Advance x by the exact rendered width of the bold prefix
                    ox += bold_font.text_length(prefix, fontsize=w.size - font_offset)
                if suffix:
                    active_writer.append((ox, oy), suffix,
                                         font=normal_font, fontsize=w.size - font_offset)
            except Exception as e:
                print("\n[ERROR] Skipped a word during rendering!")
                print(f"  -> Failed Word: '{text}'")
                print(f"  -> Error Message: {e}")
                print(f"  -> Font Used: {w.font}\n")
                # Silently skip words with unrenderable glyphs (e.g. ligatures
                # or characters not present in the fallback font). The rasterised
                # background will still show these correctly via the pixmap.
                pass

    # Flush all buffered TextWriter glyphs onto the page as a single content stream
    # writer.write_text(page)
    # 2. Render all color layers onto the page after the text loops are done
    for col, writer in writers.items():
        # Pass the unique color vector directly into the final page writer execution
        writer.write_text(page, color=col)


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    # ── 1. Define CLI arguments ───────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        prog="bionic_reader",
        description="Bionic Reader converter for research PDFs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input",  help="Source PDF path")
    parser.add_argument("output", nargs="?",
                        help="Output PDF path  [default: <input>_bionic.pdf]")
    parser.add_argument("--bold-ratio", type=float, default=0.45, metavar="RATIO",
                        help="Fraction of each word to bold  (default: 0.45)")
    parser.add_argument("--sentence-colors", type=str, default=None,
                        metavar="HEX,HEX",
                        help="Comma-separated hex colours for sentence cycling "
                             "(e.g. '#0A3D8F,#1A5C2E')")
    parser.add_argument("--pages", type=str, default=None, metavar="RANGE",
                        help="Pages to process e.g. '1-5' or '2,4,6'  "
                             "(default: all)")
    parser.add_argument("--dpi", type=int, default=150,
                        help="Background render DPI  (default: 150)")
    parser.add_argument("--no-skip-math", action="store_true",
                        help="Apply bionic to math spans too")
    args = parser.parse_args()

    # ── 2. Validate input path ────────────────────────────────────────────────
    in_path = Path(args.input)
    if not in_path.exists():
        sys.exit(f"Error: file not found — {in_path}")
    if in_path.suffix.lower() != ".pdf":
        sys.exit(f"Error: expected a .pdf file, got '{in_path.suffix}'")

    # ── 3. Derive output path ─────────────────────────────────────────────────
    # Use the explicitly provided output path, or auto-generate by appending
    # "_bionic" to the input filename stem (e.g. paper.pdf → paper_bionic.pdf).
    out_path = Path(args.output) if args.output else \
               in_path.with_stem(in_path.stem + "_bionic")

    # ── 4. Parse sentence colours ─────────────────────────────────────────────
    # User-supplied hex colours override the defaults. Any number of colours can
    # be provided; sentences cycle through them modulo len(colors).
    if args.sentence_colors:
        try:
            colors = [hex_to_rgb(c.strip())
                      for c in args.sentence_colors.split(",")]
        except Exception as exc:
            sys.exit(f"Error parsing --sentence-colors: {exc}")
    else:
        colors = DEFAULT_COLORS

    # skip_math is True by default; --no-skip-math flips it off
    skip_math = not args.no_skip_math

    # ── 5. Open the PDF ───────────────────────────────────────────────────────
    print(f"Input  : {in_path}")
    doc = fitz.open(str(in_path))
    n = len(doc)
    print(f"Pages  : {n}")

    # ── 6. Resolve page list ──────────────────────────────────────────────────
    # parse_pages() converts human-readable ranges to 0-indexed page numbers.
    # If --pages is not specified, process all pages in the document.
    page_indices = parse_pages(args.pages, n) if args.pages else list(range(n))
    print(f"Processing {len(page_indices)} page(s) …")

    # ── 7. Sentence colour counter ────────────────────────────────────────────
    # A mutable list is used instead of a plain int so that process_page() can
    # modify the counter in-place and it persists across page iterations.
    # This ensures sentence colours continue their cycle across page boundaries
    # rather than resetting to colour 0 at the start of each new page.
    sentence_idx = [0]

    # ── 8. Main processing loop ───────────────────────────────────────────────
    for pi in page_indices:
        # \r overwrites the same console line for a compact progress indicator
        print(f"  Page {pi + 1:>4} / {n}", end="\r", flush=True)
        process_page(
            page=doc[pi],
            bold_ratio=args.bold_ratio,
            colors=colors,
            sentence_idx=sentence_idx,
            skip_math=skip_math,
        )

    # ── 9. Save the modified PDF ──────────────────────────────────────────────
    # garbage=4 — remove all unreferenced objects (orphaned fonts, images, etc.)
    # deflate=True — compress all content streams with zlib
    # clean=True — normalise/canonicalise content stream operators
    print(f"\nSaving : {out_path}")
    doc.save(str(out_path), garbage=4, deflate=True, clean=True)
    doc.close()

    # ── 10. Print summary statistics ──────────────────────────────────────────
    size_mb = out_path.stat().st_size / 1_048_576
    print(f"Done!   {out_path}  ({size_mb:.1f} MB)")
    print(f"        ~{sentence_idx[0]} sentences  |  "
          f"{len(colors)}-colour cycle  |  "
          f"math-skip={'on' if skip_math else 'off'}")


if __name__ == "__main__":
    main()
