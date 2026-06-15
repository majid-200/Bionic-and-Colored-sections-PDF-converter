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
    import pymupdf as fitz  # pymupdf >= 1.24
except ImportError:
    sys.exit("PyMuPDF not found.  Run:  pip install pymupdf")

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_COLORS = [
    (0.05, 0.20, 0.55),   # deep navy
    (0.08, 0.42, 0.18),   # forest green
]

# Font names that typically carry math / symbol glyphs
MATH_FONT_FRAGMENTS = {
    "symbol", "zapf", "mt extra", "cambria math", "stix",
    "asana", "tex", "mathjax", "cmr", "cmsy", "cmex",
}

# Unicode ranges that signal mathematical content
MATH_CHAR_RE = re.compile(
    r"[∀-⋿⌀-⏿①-⑳←-⇿∀-∿⊀-⊿⋀-⋿"
    r"αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ]"
)

# Abbreviations that should NOT end a sentence
ABBREVS = frozenset({
    "fig", "figs", "eq", "eqs", "et", "al", "i.e", "e.g",
    "dr", "mr", "mrs", "ms", "prof", "vs", "approx",
    "sec", "ref", "tab", "no", "vol", "pp", "cf",
})


# ── Data classes ─────────────────────────────────────────────────────────────

class Word(NamedTuple):
    text: str          # word string
    origin: tuple      # (x, y) baseline of first char
    bbox: fitz.Rect    # tight bounding box
    size: float        # font size in pt
    font: str          # font name as stored in PDF
    is_math: bool      # heuristic: likely a math/symbol span


# ── Helpers ───────────────────────────────────────────────────────────────────

def hex_to_rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    if len(h) == 3:
        h = h[0]*2 + h[1]*2 + h[2]*2
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def parse_pages(spec: str, n: int) -> list[int]:
    """'1-5' or '1,3,7' → sorted 0-indexed list."""
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a) - 1, int(b)))
        else:
            out.add(int(part) - 1)
    return sorted(p for p in out if 0 <= p < n)


def n_bold_chars(word: str, ratio: float) -> int:
    """How many leading characters to render bold."""
    alpha = [c for c in word if c.isalpha()]
    if len(alpha) <= 1:
        return len(word)
    target = max(1, round(len(alpha) * ratio))
    count = 0
    for i, ch in enumerate(word):
        if ch.isalpha():
            count += 1
            if count >= target:
                return i + 1
    return len(word)


def is_math(span_font: str, span_text: str) -> bool:
    fl = span_font.lower()
    if any(f in fl for f in MATH_FONT_FRAGMENTS):
        return True
    if MATH_CHAR_RE.search(span_text):
        return True
    # Short tokens composed entirely of digits/operators → inline math
    stripped = span_text.strip()
    if len(stripped) <= 4 and re.fullmatch(r'[\d\+\-\*/=<>(){}|^_.,;\\ ]+', stripped):
        return True
    return False


def get_font(name: str, bold: bool = False) -> fitz.Font:
    """Return a fitz.Font, falling back to Helvetica variants."""
    if bold:
        fallback = "hebo"
    else:
        fallback = "helv"
    # Strip subset prefix: "ABCDEF+TimesNewRoman" → "TimesNewRoman"
    clean = name.split("+")[-1]
    for candidate in (clean, name, fallback):
        try:
            return fitz.Font(fontname=candidate)
        except Exception:
            pass
    return fitz.Font(fontname=fallback)


# ── Word extraction using rawdict (char-level bboxes) ─────────────────────────

def extract_words(page: fitz.Page, skip_math: bool) -> list[Word]:
    """
    Extract words with precise per-character bounding boxes.
    Each span is broken at whitespace boundaries.
    """
    words: list[Word] = []
    blocks = page.get_text("rawdict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]

    for block in blocks:
        if block["type"] != 0:   # 0 = text; 1 = image
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                font_name = span["font"]
                size      = span["size"]
                chars     = span["chars"]   # list of {c, origin, bbox}

                # Assemble full span text to check for math heuristics
                span_text = "".join(ch["c"] for ch in chars)
                math_flag = skip_math and is_math(font_name, span_text)

                # Group chars into words (split on whitespace chars)
                current_chars: list[dict] = []
                for ch in chars:
                    if ch["c"] in (" ", "\t", "\n", "\r"):
                        if current_chars:
                            _flush_word(current_chars, font_name, size,
                                        math_flag, words)
                            current_chars = []
                    else:
                        current_chars.append(ch)
                if current_chars:
                    _flush_word(current_chars, font_name, size, math_flag, words)

    return words


def _flush_word(chars: list[dict], font: str, size: float,
                math_flag: bool, out: list[Word]) -> None:
    text = "".join(ch["c"] for ch in chars)
    if not text.strip():
        return
    # Bounding box: union of all char bboxes
    x0 = min(ch["bbox"][0] for ch in chars)
    y0 = min(ch["bbox"][1] for ch in chars)
    x1 = max(ch["bbox"][2] for ch in chars)
    y1 = max(ch["bbox"][3] for ch in chars)
    bbox = fitz.Rect(x0, y0, x1, y1)
    origin = (chars[0]["origin"][0], chars[0]["origin"][1])
    out.append(Word(text=text, origin=origin, bbox=bbox,
                    size=size, font=font, is_math=math_flag))


# ── Sentence splitting ────────────────────────────────────────────────────────

def split_sentences(words: list[Word]) -> list[list[Word]]:
    """Group words into sentences using punctuation heuristics."""
    sentences: list[list[Word]] = []
    current: list[Word] = []
    for w in words:
        current.append(w)
        t = w.text
        if re.search(r'[.!?]["\')\]]?$', t):
            # Avoid splitting on "Fig.", single capitals, known abbreviations
            core = t.rstrip('.!?"\'()[]').lower()
            if core not in ABBREVS and not re.fullmatch(r'[a-z]', core):
                sentences.append(current)
                current = []
    if current:
        sentences.append(current)
    return sentences


# ── Page processing ───────────────────────────────────────────────────────────

def process_page(page: fitz.Page, bold_ratio: float, colors: list,
                 sentence_idx: list[int], skip_math: bool) -> None:
    """
    Overlay bionic reading + sentence colouring onto *page* (modified in-place).

    Strategy:
      1. Render original page to a pixmap at moderate DPI.
      2. Blank the page and insert the pixmap as background image
         (this locks in figures, equations, everything visual).
      3. Lay new bionic text on top using TextWriter.
    """
    # ── Step 1: render original to image ─────────────────────────────────────
    mat = fitz.Matrix(1.5, 1.5)   # 1.5× → ~108 dpi at 72 pt/inch
    pix = page.get_pixmap(matrix=mat, alpha=False)

    # ── Step 2: collect words BEFORE modifying page ───────────────────────────
    words = extract_words(page, skip_math)
    sentences = split_sentences(words)

    # ── Step 3: blank the page, reinsert background image ────────────────────
    page.clean_contents()
    # Remove all existing content streams (text + graphics)
    # Keep page intact by inserting bg image first
    rect = page.rect
    page.insert_image(rect, pixmap=pix, keep_proportion=False)

    # ── Step 4: draw semi-transparent white knockout behind text regions ──────
    shape = page.new_shape()
    for sent in sentences:
        for w in sent:
            if w.is_math:
                continue
            pad = 1.2
            r = fitz.Rect(w.bbox.x0 - pad, w.bbox.y0 - pad,
                           w.bbox.x1 + pad, w.bbox.y1 + pad)
            shape.draw_rect(r)
            shape.finish(fill=(1, 1, 1), fill_opacity=0.88,
                         color=None, width=0)
    shape.commit()

    # ── Step 5: write bionic text ─────────────────────────────────────────────
    writer = fitz.TextWriter(rect)

    for sent in sentences:
        col = colors[sentence_idx[0] % len(colors)]
        sentence_idx[0] += 1

        for w in sent:
            if w.is_math:
                continue
            text = w.text
            if not text.strip():
                continue

            nb = n_bold_chars(text, bold_ratio)
            prefix = text[:nb]
            suffix  = text[nb:]

            bold_font   = get_font(w.font, bold=True)
            normal_font = get_font(w.font, bold=False)

            ox, oy = w.origin
            try:
                if prefix:
                    writer.append((ox, oy), prefix,
                                  font=bold_font, fontsize=w.size, color=col)
                    ox += bold_font.text_length(prefix, fontsize=w.size)
                if suffix:
                    writer.append((ox, oy), suffix,
                                  font=normal_font, fontsize=w.size, color=col)
            except Exception:
                pass   # silently skip unrenderable glyphs (e.g. ligatures)

    writer.write_text(page)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
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

    in_path = Path(args.input)
    if not in_path.exists():
        sys.exit(f"Error: file not found — {in_path}")
    if in_path.suffix.lower() != ".pdf":
        sys.exit(f"Error: expected a .pdf file, got '{in_path.suffix}'")

    out_path = Path(args.output) if args.output else \
               in_path.with_stem(in_path.stem + "_bionic")

    # Colours
    if args.sentence_colors:
        try:
            colors = [hex_to_rgb(c.strip())
                      for c in args.sentence_colors.split(",")]
        except Exception as exc:
            sys.exit(f"Error parsing --sentence-colors: {exc}")
    else:
        colors = DEFAULT_COLORS

    skip_math = not args.no_skip_math

    print(f"Input  : {in_path}")
    doc = fitz.open(str(in_path))
    n = len(doc)
    print(f"Pages  : {n}")

    page_indices = parse_pages(args.pages, n) if args.pages else list(range(n))
    print(f"Processing {len(page_indices)} page(s) …")

    sentence_idx = [0]   # mutable box so it carries across pages

    for pi in page_indices:
        print(f"  Page {pi + 1:>4} / {n}", end="\r", flush=True)
        process_page(
            page=doc[pi],
            bold_ratio=args.bold_ratio,
            colors=colors,
            sentence_idx=sentence_idx,
            skip_math=skip_math,
        )

    print(f"\nSaving : {out_path}")
    doc.save(str(out_path), garbage=4, deflate=True, clean=True)
    doc.close()

    size_mb = out_path.stat().st_size / 1_048_576
    print(f"Done!   {out_path}  ({size_mb:.1f} MB)")
    print(f"        ~{sentence_idx[0]} sentences  |  "
          f"{len(colors)}-colour cycle  |  "
          f"math-skip={'on' if skip_math else 'off'}")


if __name__ == "__main__":
    main()
