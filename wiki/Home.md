# CCC 2026 Knowledge Base

Team wiki for the **42nd Cloudflight Coding Contest (CCC42)**, **#data/ai track**, **Fri 23 Oct 2026, 13:00 UTC, 4 hours**.

The official track description says: *"Whether you're exploring machine learning models or **experimenting with neural networks**, this category is designed to put your skills to the test."* So expect at least one level where a neural network is the intended (or the easiest) route. Don't assume every level needs one, though: the 2025 winners used only pandas and sklearn.

## Pages

| Page | Read it when |
|---|---|
| [Contest Format](Contest-Format) | Before the contest: rules, files, scoring |
| [Challenge Workflow](Challenge-Workflow) | At the start of every level, step by step |
| [Data Cleaning Checklist](Data-Cleaning-Checklist) | First 10 minutes of every data level |
| [Validation and Metrics](Validation-and-Metrics) | Before trusting any score |
| [Classic ML Playbook](Classic-ML-Playbook) | Regression, random forests, boosting |
| [Neural Networks Playbook](Neural-Networks-Playbook) | **Main NN page**: PyTorch recipes, tuning, debugging |
| [NN Architectures by Data Type](NN-Architectures-by-Data-Type) | Tabular, sequence, image, audio, time series |
| [Neural Networks from Scratch](Neural-Networks-from-Scratch) | Given weights, "implement the forward pass", backprop by hand |
| [Past Challenges](Past-Challenges) | What the 2022/2023/2025 AI rounds looked like |
| [Pitfalls](Pitfalls) | Before every upload |
| [Environment Setup](Environment-Setup) | Setting up a laptop |

## Quick start on contest day

```bash
git clone https://github.com/<owner>/ccc-2026 && cd ccc-2026
uv sync                      # Python 3.14 + all packages (CUDA torch)
uv run ccc gpu               # check the GPU is visible
uv run ccc new 1             # levels/level1/ from template
uv run ccc unzip ~/Downloads/level_1.zip 1
uv run python levels/level1/solve.py
```
