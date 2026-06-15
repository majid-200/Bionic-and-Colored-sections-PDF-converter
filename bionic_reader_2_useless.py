"""
Bionic Reader for Research Papers
===================================
Converts a PDF into a Bionic Reading format:
  - Bolds the first ~45% of each word (configurable)
  - Colours each sentence alternately so you can visually track start → end
  - Preserves ALL figures, equations, vector drawings, and images exactly
  - Handles PDFs with broken xrefs (common in arXiv/conference papers)

Strategy:
  1. Repair the PDF in-memory (fixes xref corruption)
  2. Extract word positions via char-level rawdict
  3. Redact non-math words with white fill (surgically removes only text)
  4. Apply redactions — preserves all vector drawings and raster images
  5. Write new bionic-coloured text via per-colour TextWriter instances

Requirements:
    pip install pymupdf

Usage (Windows terminal):
    python bionic_reader.py paper.pdf
    python bionic_reader.py paper.pdf output.pdf
    python bionic_reader.py paper.pdf --bold-ratio 0.5 --pages 1-10
    python bionic_reader.py paper.pdf --sentence-colors "#0A3D8F,#8B0000,#1A5C2E"
    python bionic_reader.py paper.pdf --no-skip-math

Options:
    input                   Source PDF path
    output                  Output PDF path  [default: <input>_bionic.pdf]
    --bold-ratio FLOAT      Fraction of each word to bold  [default: 0.45]
    --sentence-colors HEX   Comma-separated hex colours for sentence cycling
    --pages RANGE           Pages to process e.g. "1-5" or "2,4,6"  [default: all]
    --no-skip-math          Apply bionic to math/symbol spans too
"""

from __future__ import annotations
import argparse
import io
import re
import sys
from pathlib import Path
from typing import NamedTuple

try:
    import pymupdf as fitz
except ImportError:
    sys.exit("pymupdf not found.  Run:  pip install pymupdf")

# ── Colour defaults ───────────────────────────────────────────────────────────
DEFAULT_COLORS = [
    (0.05, 0.20, 0.55),   # deep navy
    (0.08, 0.42, 0.18),   # forest green
]

# Font name fragments that indicate math/symbol glyphs
MATH_FONT_FRAGMENTS = {
    "cmsy", "cmex", "cmmi", "msam", "msbm",
    "symbol", "zapf", "mt extra", "cambria math", "stix",
    "asana", "mathjax", "eulervm", "fourier",
}

# Unicode ranges that signal mathematical content
MATH_CHAR_RE = re.compile(
    r"[∀-⋿⌀-⏿←-⇿∀-∿⊀-⊿⋀-⋿"
    r"αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ"
    r"∑∏∫∂∇√∞≈≠≤≥±×÷]"
)

# Abbreviations that should NOT end a sentence
ABBREVS = frozenset({
    "fig", "figs", "eq", "eqs", "et", "al", "i.e", "e.g",
    "dr", "mr", "mrs", "ms", "prof", "vs", "approx",
    "sec", "ref", "tab", "no", "vol", "pp", "cf", "proc",
    "dept", "univ", "est", "prob", "avg", "max", "min",
})


# ── Data types ────────────────────────────────────────────────────────────────

class Word(NamedTuple):
    text: str
    origin: tuple        # (x, y) baseline of first character
    bbox: fitz.Rect      # tight bounding box over all chars
    size: float          # font size in points
    font: str            # font name as stored in PDF
    flags: int           # span flags (bold=16, italic=2, etc.)
    is_math: bool        # heuristic: likely a math/symbol span


# ── Helpers ───────────────────────────────────────────────────────────────────

def hex_to_rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    if len(h) == 3:
        h = h[0]*2 + h[1]*2 + h[2]*2
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def parse_pages(spec: str, n: int) -> list[int]:
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
    """Return how many leading characters of `word` to render bold."""
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


def is_math_span(font: str, text: str) -> bool:
    fl = font.lower().split("+")[-1]   # strip subset prefix
    if any(f in fl for f in MATH_FONT_FRAGMENTS):
        return True
    if MATH_CHAR_RE.search(text):
        return True
    stripped = text.strip()
    if 1 <= len(stripped) <= 4 and re.fullmatch(r'[\d\+\-\*/=<>(){}|^_.,;\\ ]+', stripped):
        return True
    return False


def get_font(name: str, bold: bool = False) -> fitz.Font:
    """Return a pymupdf Font, falling back to Helvetica variants."""
    fallback = "hebo" if bold else "helv"
    clean = name.split("+")[-1]
    for candidate in (clean, name, fallback):
        try:
            return fitz.Font(fontname=candidate)
        except Exception:
            pass
    return fitz.Font(fontname=fallback)


# ── Word extraction ───────────────────────────────────────────────────────────

def extract_words(page: fitz.Page, skip_math: bool) -> list[Word]:
    words: list[Word] = []
    try:
        blocks = page.get_text(
            "rawdict",
            flags=fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_PRESERVE_LIGATURES
        )["blocks"]
    except Exception:
        return words

    for block in blocks:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                font_name = span.get("font", "")
                size = span.get("size", 10.0)
                flags = span.get("flags", 0)
                chars = span.get("chars", [])
                if not chars:
                    continue

                span_text = "".join(ch["c"] for ch in chars)
                math_flag = skip_math and is_math_span(font_name, span_text)

                current: list[dict] = []
                for ch in chars:
                    if ch["c"] in (" ", "\t", "\n", "\r", "\x0c"):
                        if current:
                            _flush_word(current, font_name, size, flags, math_flag, words)
                            current = []
                    else:
                        current.append(ch)
                if current:
                    _flush_word(current, font_name, size, flags, math_flag, words)

    return words


def _flush_word(chars: list[dict], font: str, size: float,
                flags: int, math_flag: bool, out: list[Word]) -> None:
    text = "".join(ch["c"] for ch in chars)
    if not text.strip():
        return
    x0 = min(ch["bbox"][0] for ch in chars)
    y0 = min(ch["bbox"][1] for ch in chars)
    x1 = max(ch["bbox"][2] for ch in chars)
    y1 = max(ch["bbox"][3] for ch in chars)
    bbox = fitz.Rect(x0, y0, x1, y1)
    origin = (chars[0]["origin"][0], chars[0]["origin"][1])
    out.append(Word(text=text, origin=origin, bbox=bbox,
                    size=size, font=font, flags=flags, is_math=math_flag))


# ── Sentence splitting ────────────────────────────────────────────────────────

def split_sentences(words: list[Word]) -> list[list[Word]]:
    sentences: list[list[Word]] = []
    current: list[Word] = []
    for w in words:
        current.append(w)
        t = w.text
        if re.search(r'[.!?]["\')\]]?$', t):
            core = t.rstrip('.!?"\'()[]').lower()
            if core not in ABBREVS and not re.fullmatch(r'[a-z]', core) and len(core) > 1:
                sentences.append(current)
                current = []
    if current:
        sentences.append(current)
    return sentences


# ── Page processing ───────────────────────────────────────────────────────────

def process_page(page: fitz.Page, bold_ratio: float,
                 colors: list, sentence_idx: list[int],
                 skip_math: bool) -> None:
    """
    Apply bionic reading + sentence colouring to a page in-place.

    Steps:
      1. Extract words with char-level bboxes.
      2. Add white redact annotations over each non-math word.
      3. Apply redactions — removes text only, keeps vector drawings/images.
      4. Use one TextWriter per sentence colour, write all text per colour.
    """

    # ── 1. Extract words ──────────────────────────────────────────────────────
    words = extract_words(page, skip_math)
    if not words:
        return
    sentences = split_sentences(words)

    # ── 2 & 3. Redact non-math words ─────────────────────────────────────────
    PAD = 0.5
    for sent in sentences:
        for w in sent:
            if w.is_math:
                continue
            r = fitz.Rect(w.bbox.x0 - PAD, w.bbox.y0 - PAD,
                           w.bbox.x1 + PAD, w.bbox.y1 + PAD)
            if r.is_valid and not r.is_empty:
                page.add_redact_annot(r, fill=(1, 1, 1))

    page.apply_redactions(
        images=fitz.PDF_REDACT_IMAGE_NONE,
        graphics=fitz.PDF_REDACT_LINE_ART_NONE,
    )

    # ── 4. Write bionic text — one TextWriter per unique colour ───────────────
    # Group (sentence, colour_index) then write per colour
    n_colors = len(colors)

    # Build per-colour writers
    writers: dict[int, fitz.TextWriter] = {
        i: fitz.TextWriter(page.rect) for i in range(n_colors)
    }

    bold_font_cache: dict[str, fitz.Font] = {}
    norm_font_cache: dict[str, fitz.Font] = {}

    def get_cached_font(name: str, bold: bool) -> fitz.Font:
        cache = bold_font_cache if bold else norm_font_cache
        if name not in cache:
            cache[name] = get_font(name, bold=bold)
        return cache[name]

    for sent in sentences:
        cidx = sentence_idx[0] % n_colors
        sentence_idx[0] += 1
        writer = writers[cidx]

        for w in sent:
            if w.is_math:
                continue
            text = w.text
            if not text.strip():
                continue

            nb = n_bold_chars(text, bold_ratio)
            prefix = text[:nb]
            suffix = text[nb:]

            is_orig_bold = bool(w.flags & 2**4)
            bf = get_cached_font(w.font, bold=True)
            nf = get_cached_font(w.font, bold=is_orig_bold)

            ox, oy = w.origin
            try:
                if prefix:
                    writer.append((ox, oy), prefix, font=bf, fontsize=w.size)
                    ox += bf.text_length(prefix, fontsize=w.size)
                if suffix:
                    writer.append((ox, oy), suffix, font=nf, fontsize=w.size)
            except Exception:
                pass

    # Write each colour group to the page
    for cidx, writer in writers.items():
        if writer.text_rect.is_empty:
            continue
        writer.write_text(page, color=colors[cidx])


# ── Main ──────────────────────────────────────────────────────────────────────

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
                        help="Comma-separated hex colours for sentence cycling")
    parser.add_argument("--pages", type=str, default=None, metavar="RANGE",
                        help="Pages to process e.g. '1-5' or '2,4,6'  (default: all)")
    parser.add_argument("--no-skip-math", action="store_true",
                        help="Apply bionic to math/symbol spans too")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        sys.exit(f"Error: file not found — {in_path}")
    if in_path.suffix.lower() != ".pdf":
        sys.exit(f"Error: expected a .pdf file, got '{in_path.suffix}'")

    out_path = Path(args.output) if args.output else \
               in_path.with_stem(in_path.stem + "_bionic")

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
    print("Opening and repairing PDF …")

    # ── Repair broken xrefs by round-tripping through a buffer ───────────────
    raw_doc = fitz.open(str(in_path))
    buf = io.BytesIO()
    raw_doc.save(buf, garbage=4, deflate=True, clean=True)
    raw_doc.close()
    buf.seek(0)
    doc = fitz.open("pdf", buf.read())

    n = len(doc)
    print(f"Pages  : {n}")

    page_indices = parse_pages(args.pages, n) if args.pages else list(range(n))
    print(f"Processing {len(page_indices)} page(s)  |  "
          f"bold-ratio={args.bold_ratio}  |  "
          f"math-skip={'on' if skip_math else 'off'}")

    sentence_idx = [0]

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
    doc.save(str(out_path), garbage=4, deflate=True)
    doc.close()

    size_mb = out_path.stat().st_size / 1_048_576
    print(f"Done!   {out_path}  ({size_mb:.1f} MB)")
    print(f"        ~{sentence_idx[0]} sentences  |  {len(colors)}-colour cycle  |  "
          f"math-skip={'on' if skip_math else 'off'}")


if __name__ == "__main__":
    main()
