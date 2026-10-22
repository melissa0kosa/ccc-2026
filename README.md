# ccc-2026

Team workspace for the **42nd Cloudflight Coding Contest**, Data & AI track (23 Oct 2026).

- **Wiki:** see the repo wiki, or `wiki/` in this repo. Start at `wiki/Home.md`.
- **Claude Code instructions:** `CLAUDE.md`
- **Toolkit:** `src/ccc_2026/` (I/O, CV harness, PyTorch trainer, NumPy NN, PDF/audio/video intake)

```bash
uv sync                                # Python 3.14 + ML stack (CUDA 13 torch)
uv run ccc gpu                         # check GPU
uv run pytest -q                       # self-test
uv run ccc new 1                       # levels/level1 from template
uv run ccc unzip ~/Downloads/level_1.zip 1
uv run python levels/level1/solve.py
```

No NVIDIA GPU? Remove the `[[tool.uv.index]]` and `[tool.uv.sources]` blocks from `pyproject.toml`, then run `uv lock && uv sync`.
