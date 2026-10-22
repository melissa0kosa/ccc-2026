"""Turning non-PDF challenge statements into text: PDFs, audio and video.

    uv run python -m ccc_2026.media pdf  "levels/level1/Level 1.pdf"
    uv run python -m ccc_2026.media transcribe levels/level1/statement.mp4
    uv run python -m ccc_2026.media download <video-url> levels/level1/
"""

from __future__ import annotations

import sys
from pathlib import Path


def pdf_to_text(pdf: str | Path) -> Path:
    """Extract text page by page into <pdf>.txt (slides keep their page separators)."""
    import pymupdf

    pdf = Path(pdf)
    with pymupdf.open(pdf) as doc:
        parts = [f"--- page {i + 1} ---\n{page.get_text()}" for i, page in enumerate(doc)]
    out = pdf.with_suffix(".txt")
    out.write_text("\n".join(parts), encoding="utf-8")
    return out


def pdf_to_images(pdf: str | Path, dpi: int = 110) -> list[Path]:
    """Render each page to PNG (for slides that are image-only or contain diagrams/tables)."""
    import pymupdf

    pdf = Path(pdf)
    out_dir = pdf.with_suffix("")
    out_dir.mkdir(exist_ok=True)
    paths = []
    with pymupdf.open(pdf) as doc:
        for i, page in enumerate(doc):
            p = out_dir / f"page_{i + 1:02d}.png"
            page.get_pixmap(dpi=dpi).save(p)
            paths.append(p)
    return paths


def transcribe(media: str | Path, model: str = "small", language: str | None = "en") -> Path:
    """Speech-to-text with Whisper (needs ffmpeg on PATH). Writes <media>.transcript.txt
    with [mm:ss] timestamps so numbers/rules can be traced back to the recording."""
    import torch
    import whisper

    media = Path(media)
    m = whisper.load_model(model, device="cuda" if torch.cuda.is_available() else "cpu")
    result = m.transcribe(str(media), language=language)
    lines = [f"[{int(s['start']) // 60:02d}:{int(s['start']) % 60:02d}] {s['text'].strip()}"
             for s in result["segments"]]
    out = media.with_suffix(".transcript.txt")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def download(url: str, dest: str | Path) -> None:
    """Download a video/audio statement (YouTube, Vimeo, direct links...) with yt-dlp,
    including subtitles if the platform provides them."""
    import yt_dlp

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    opts = {"outtmpl": str(dest / "statement.%(ext)s"), "writesubtitles": True,
            "writeautomaticsub": True, "subtitleslangs": ["en.*"]}
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    cmd, arg = sys.argv[1], sys.argv[2]
    if cmd == "pdf":
        print(pdf_to_text(arg))
        print(*pdf_to_images(arg), sep="\n")
    elif cmd == "transcribe":
        print(transcribe(arg, *(sys.argv[3:4] or ["small"])))
    elif cmd == "download":
        download(arg, sys.argv[3] if len(sys.argv) > 3 else ".")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
