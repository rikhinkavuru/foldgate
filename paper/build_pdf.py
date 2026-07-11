#!/usr/bin/env python3
"""Build a clean PDF from the markdown paper.

pandoc's default LaTeX font (Latin Modern) lacks Greek letters and math symbols,
so we convert those Unicode characters to inline LaTeX math before rendering.
The source .md keeps the readable Unicode; this only affects the PDF build.

Usage: python paper/build_pdf.py paper/moml2026_shortpaper.md
"""

import re
import subprocess
import sys
import tempfile
from pathlib import Path

MATH = {
    "α": r"\alpha", "τ": r"\tau", "δ": r"\delta", "β": r"\beta",
    "≤": r"\le", "≥": r"\ge", "→": r"\to", "×": r"\times",
    "≈": r"\approx", "≠": r"\ne",
}
PLAIN = {"−": "-", "–": "--", "—": "---", "Å": r"\AA{}", "Δ": r"$\Delta$"}


def convert(text: str) -> str:
    for ch, tex in MATH.items():
        text = text.replace(ch, f"${tex}$")
    for ch, rep in PLAIN.items():
        text = text.replace(ch, rep)
    # collapse accidental adjacent math like $\le$$\alpha$ is fine for pandoc
    return text


def main() -> None:
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "paper/moml2026_shortpaper.md")
    out = src.with_suffix(".pdf")
    body = convert(src.read_text())
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write(body)
        tmp = f.name
    cmd = ["pandoc", tmp, "-o", str(out), "--pdf-engine=tectonic",
           "-V", "geometry:margin=1in", "-V", "fontsize=10pt"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    warn = [ln for ln in r.stderr.splitlines() if "Missing character" in ln or "could not represent" in ln]
    if r.returncode == 0 and not warn:
        print(f"wrote {out} ({out.stat().st_size} bytes), no missing glyphs")
    else:
        print(f"pandoc rc={r.returncode}; {len(warn)} glyph warnings")
        print("\n".join(r.stderr.splitlines()[-8:]))


if __name__ == "__main__":
    main()
