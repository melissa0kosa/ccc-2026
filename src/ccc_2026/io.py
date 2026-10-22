"""Reading CCC input files and writing output files in the exact expected format."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def extract_zip(zip_path: str | Path, dest: str | Path | None = None) -> Path:
    """Extract a level zip into its own folder (default: next to the zip, same stem)."""
    zip_path = Path(zip_path)
    dest = Path(dest) if dest else zip_path.with_suffix("")
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            target = (dest / member).resolve()
            if not target.is_relative_to(dest.resolve()):
                raise ValueError(f"unsafe path in zip: {member}")
        zf.extractall(dest)
    return dest


def list_inputs(level_dir: str | Path) -> list[Path]:
    """All input files of a level, sorted; sample/example files first."""
    files = [p for p in Path(level_dir).rglob("*") if p.suffix in {".in", ".txt", ".csv"}]
    files = [p for p in files if not _is_output(p)]
    return sorted(files, key=lambda p: (not is_sample(p), p.name))


def is_sample(path: Path) -> bool:
    name = path.name.lower()
    return "sample" in name or "example" in name


def _is_output(path: Path) -> bool:
    return path.suffix == ".out" or path.stem.endswith(".out")


def sample_output_for(input_path: Path) -> Path | None:
    """Matching expected-output file for a sample input, if present."""
    for cand in (input_path.with_suffix(".out"), input_path.with_name(input_path.name + ".out")):
        if cand.exists():
            return cand
    return None


def read_lines(path: str | Path) -> list[str]:
    return Path(path).read_text(encoding="utf-8-sig").splitlines()


def read_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    """Read a CSV-style .in file. Sniffs the separator (comma / semicolon / whitespace)."""
    text = Path(path).read_text(encoding="utf-8-sig")
    first = text.splitlines()[0] if text else ""
    if "," in first:
        sep = ","
    elif ";" in first:
        sep = ";"
    else:
        sep = r"\s+"
    return pd.read_csv(io.StringIO(text), sep=sep, engine="python" if sep == r"\s+" else "c",
                       **kwargs)


def output_path(input_path: str | Path, out_dir: str | Path | None = None) -> Path:
    input_path = Path(input_path)
    out_dir = Path(out_dir) if out_dir else input_path.parent.parent / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / (input_path.stem + ".out")


def write_lines(path: str | Path, lines) -> Path:
    path = Path(path)
    path.write_text("\n".join(str(x) for x in lines) + "\n", encoding="utf-8")
    return path


def write_csv(path: str | Path, df: pd.DataFrame, header: bool = True) -> Path:
    """Write CSV with '\\n' line endings and no index (Windows would otherwise add '\\r\\n')."""
    path = Path(path)
    df.to_csv(path, index=False, header=header, lineterminator="\n")
    return path


def fmt(x, decimals: int | None = None) -> str:
    """Format a scalar without numpy noise (np.float64(1.0) -> '1.0', fixed decimals optional)."""
    if isinstance(x, (np.integer, int)):
        return str(int(x))
    if decimals is not None:
        return f"{float(x):.{decimals}f}"
    return repr(float(x)) if isinstance(x, (np.floating, float)) else str(x)


def compare_to_sample(produced: str | Path, expected: str | Path, tol: float = 1e-6) -> list[str]:
    """Token-wise diff of two output files (numbers compared with tolerance). Empty list = match."""
    got, exp = read_lines(produced), read_lines(expected)
    problems: list[str] = []
    if len(got) != len(exp):
        problems.append(f"line count differs: got {len(got)}, expected {len(exp)}")
    for i, (g, e) in enumerate(zip(got, exp, strict=False), start=1):
        if g.strip() == e.strip():
            continue
        gt, et = g.replace(",", " ").split(), e.replace(",", " ").split()
        if len(gt) == len(et) and all(_tok_eq(a, b, tol) for a, b in zip(gt, et, strict=True)):
            continue
        problems.append(f"line {i}: got {g!r}, expected {e!r}")
        if len(problems) > 10:
            problems.append("...")
            break
    return problems


def _tok_eq(a: str, b: str, tol: float) -> bool:
    if a == b:
        return True
    try:
        return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))
    except ValueError:
        return False
