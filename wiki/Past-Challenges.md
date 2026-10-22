# Past AI / Data Challenges

## 2025: CCC41 "Migratory Birds" (Data & AI, 6 levels)
Official PDFs and zips: https://github.com/goncalogalante/CCContest-2025-AI-solution. The game is playable at https://codingcontest.org/challenges.

| Level | Task | Winning approach | Pass criterion |
|---|---|---|---|
| L1 | Sort locations by temperature → humidity → id | Data cleaning: spelled-out temps, °F mixed with °C | Exact |
| L2 | Predict the missing "Bird Love Score" | Fix units; **linear regression** | RMSE < 6 (a), < 3 (b, c) |
| L3 | Match 6 descriptions to 6 flock path patterns | Rules | Exact |
| L4 | Classify flocks into 6 species from path sequences | One feature row per flock + **random forest** | Macro-F1 > 0.90 |
| L5 | Top-50 locations by arrivals, for each of the next 30 days | Classifier ranking; validate on the **most recent** windows | Mean daily accuracy ≥ 50% |
| L6 | Predict one day | Data was a **time-reversed mirror** of L5, so it was a lookup | 100% |

**Lessons:** no neural network was needed. Clean data and spotting hidden structure beat modelling.

## 2023: AI category (11 levels)
Writeup: https://github.com/sheilateozy/cloudflight-ai-coding-contest
- **L1:** PCA on a CSV; report the std of 2 components to 2 decimals.
- **L2:** scatter-plot the result; the picture shows a **password** that unlocks the next zip.
- **L3:** drop NaN and out-of-range rows; report the count.
- **L4:** Kelvin → Celsius; infer the "?" units from min/max.
- **L5–L6:** tabular regression, RMSE ≤ 2.0, then ≤ 1.5.
- **L7:** 2500 predictions reshaped to **50×50** form an image with the password.
- **L8–L11:** RMSE ≤ 1.0 / 0.75 / 0.5 / 0.45. The top-10% solution was a stacked DNN + XGBoost + LightGBM ensemble.

## 2022: AI (8 levels)
Repo: https://github.com/MattiaSangermano/Cloudflight-Coding-Contest-AI-2022
- **Sequence classification:** 30-character strings (a–i) → 5 classes, with an imbalanced test set.
- **Solution:** Keras LSTM(256) → LSTM(128) → Dense, with RandomOverSampler and a stratified split, reaching about 0.99 macro-F1. This is the clearest past case where a **neural network** was the natural tool.

## Recurring patterns
1. Messy data: units, spelled-out numbers, NaN, outliers.
2. Exact-answer analysis levels before the modelling levels.
3. Thresholded metrics (RMSE, macro-F1, accuracy) that get stricter level by level.
4. **Hidden puzzles:** plot it, reshape it to a square image, look for mirrors or leaks.
5. Data carries forward between levels.

## Practice
Play CCC41 "Migratory Birds" on https://codingcontest.org/challenges with this repo's workflow as a dry run.
