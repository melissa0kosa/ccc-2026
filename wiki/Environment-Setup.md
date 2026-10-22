# Environment Setup

## Requirements
- Git
- [uv](https://docs.astral.sh/uv/). Windows: `winget install astral-sh.uv`. Mac/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- NVIDIA GPU with driver ≥ 580 for CUDA 13 wheels (optional)

## Install
```bash
git clone https://github.com/melissa0kosa/ccc-2026 && cd ccc-2026
uv sync            # installs Python 3.14 if needed and creates .venv
uv run ccc gpu     # torch / CUDA status
uv run pytest -q   # toolkit self-test
```

## What's installed
| Area | Packages |
|---|---|
| Data | numpy, pandas, polars, pyarrow, scipy |
| Classic ML | scikit-learn, xgboost, lightgbm, catboost, optuna, shap |
| Neural nets | torch, torchvision, torchaudio (**cu130** wheels for RTX 50xx / Blackwell) |
| Media / intake | pillow, opencv-python, soundfile, librosa, pymupdf (PDF → text + page PNGs), yt-dlp (video download), openai-whisper (speech→text; needs ffmpeg: `winget install Gyan.FFmpeg`) |
| Notebook/plots | jupyterlab, ipykernel, ipywidgets, matplotlib, seaborn |
| Misc | tqdm, rich, numba, sympy, networkx |
| Dev (group) | ruff, pytest, ipython, mypy |

Torch is pinned to the PyTorch `cu130` index in `pyproject.toml` (`[tool.uv.sources]`). On a laptop with **no NVIDIA GPU**, delete the `[[tool.uv.index]]` and `[tool.uv.sources]` blocks and run `uv lock && uv sync` to get CPU wheels. Do **not** use the cu126 index on RTX 50xx; it lacks sm_120 kernels.

## Day-before checklist
- [ ] `uv sync` done on every laptop, with no downloads needed on site.
- [ ] Whisper model cached: `uv run python -c "import whisper; whisper.load_model('small')"`
- [ ] Pretrained weights cached, if you might need them: `uv run python -c "import torchvision; torchvision.models.resnet18(weights='DEFAULT')"`
- [ ] Logged into the new codingcontest.org platform, with the team created.
- [ ] Practice game played (CCC41 Migratory Birds).
- [ ] Jupyter works: `uv run jupyter lab`

## Useful commands
```bash
uv add <pkg>                  # add a dependency (commit pyproject.toml + uv.lock)
uv run jupyter lab
uv run ruff check . --fix
uv run python levels/level3/solve.py
```
