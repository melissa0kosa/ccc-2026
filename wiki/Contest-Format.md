# Contest Format

Labels: **verified** means checked on the official site in Oct 2026. *Unconfirmed* means seen in older editions or secondary sources only.

## CCC42 (2026) facts (verified)
- **Date:** Friday 23 Oct 2026.
- **Start times:** #school 08:00 UTC, #classic 13:00 UTC, **#data/ai 13:00 UTC**. Central Europe is still on summer time (CEST) on the 23rd, so 13:00 UTC is **15:00 local** in Vienna and Berlin, and **16:00** in Cluj and Bucharest.
- **Duration:** classic 4 h, **data/ai 4 h**, school 2 h.
- **Teams:** solo or **2–3 people** (classic, data/ai). Every member can submit, and you may use several laptops.
- **Ranking:** *the candidate (or team) reaching the highest level within the shortest time wins*. Speed matters as well as correctness.
- **Languages:** all are allowed ("any technology you love").
- **Platform:** a new one in 2026 (codingcontest.org, replacing CatCoder). Log in **before** the start and explore the UI. Past games are playable for practice at https://codingcontest.org/challenges.
- **Expected size:** about 6 levels ("Six levels almost nobody finishes").
- **On-site rule:** no excessive downloads or streaming. **Pre-download** wheels and pretrained weights the day before.
- **AI/LLM tools:** **no rule found** in How-it-works or the Terms. Check the contest-day statement or Discord (https://discord.gg/eJKvPJW7au) to be sure.

## How a level is delivered (from the 2025 Data & AI round)
- `Level N.pdf`: a slide-deck problem statement. Always read the **"Acceptance Criteria"** slide; it gives the metric and the threshold.
- `level_N.zip` contains:
  - `level_N_sample.in` and `level_N_sample.out`, a worked example
  - `level_N_a.in` … `level_N_e.in`, the inputs you must solve (one output per input)
  - sometimes extra files (`level_3_species.in`, `all_data_from_level_1.in`)
- Inputs are usually CSV with no spaces after commas. Some levels have **one huge file** (2025 L5: 58 MB, ~1.9 M rows).
- You **upload one output file (txt/csv) per input**. Each is graded pass/fail against a threshold (exact, RMSE < x, macro-F1 > y, accuracy ≥ z).
- The next level unlocks once the current one passes. Later levels often **reuse earlier data**, so keep the parsing/cleaning code reusable.

## Scoring details (*unconfirmed for 2026*, true in 2023)
- **+1 penalty minute** per wrong submission, so validate locally before uploading.
- **−2 bonus minutes** for uploading source code after finishing a level. Always upload the code.

## Statement formats this year
The team has heard that statements might come as **video or audio** this year. That is not confirmed; every past statement was a PDF. If it happens:
1. Open the contest page in a browser (Claude can use Playwright).
2. Save the media and any visible text or attachments.
3. Transcribe: `uv run python -m ccc_2026.media transcribe <file>` (Whisper, timestamps), or use the platform's captions. Write `STATEMENT.md` before coding.

See also [Challenge Workflow](Challenge-Workflow).
