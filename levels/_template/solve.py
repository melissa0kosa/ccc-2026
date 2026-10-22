"""Level N solution. Run from repo root:  uv run python levels/levelN/solve.py

1. Fill in TASK / METRIC from the PDF's "Acceptance Criteria" slide.
2. Make the sample input reproduce the sample output exactly (`check` below).
3. Only then run on all inputs and upload the files in out/.
"""

from pathlib import Path

from ccc_2026.io import (
    compare_to_sample,
    is_sample,
    list_inputs,
    output_path,
    read_csv,
    sample_output_for,
    write_lines,
)

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "out"


def solve(path: Path) -> list[str]:
    df = read_csv(path)
    # TODO: parse, clean, model, predict
    return [str(len(df))]


def main() -> None:
    for inp in list_inputs(DATA):
        out = write_lines(output_path(inp, OUT), solve(inp))
        msg = f"{inp.name} -> {out.relative_to(HERE)}"
        if is_sample(inp) and (expected := sample_output_for(inp)):
            problems = compare_to_sample(out, expected)
            msg += "  SAMPLE OK" if not problems else "  SAMPLE MISMATCH:\n    " + "\n    ".join(
                problems)
        print(msg)


if __name__ == "__main__":
    main()
