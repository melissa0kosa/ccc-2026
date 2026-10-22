# Pitfalls (check before every upload)

## Output format
- [ ] Rows are in the **same order and with the same ids** as the input (or as the statement specifies).
- [ ] Header present or absent **exactly like the sample**.
- [ ] Separator matches (comma vs space vs newline). No trailing spaces.
- [ ] No numpy noise in text (`np.float64(1.5)`, `[1.5]`). Use `ccc_2026.io.fmt`.
- [ ] Decimals and rounding as specified. Integers written without `.0` if the expected output is integer.
- [ ] Line endings are `\n`. `write_csv` / `write_lines` handle this; pandas on Windows otherwise writes `\r\n`.
- [ ] One output file per input file (all of a–e), each with the right name.
- [ ] No NaN in the predictions: `assert np.isfinite(pred).all()`.

## Modelling
- [ ] Target transform inverted (`expm1`, un-standardize).
- [ ] `model.eval()` / `torch.no_grad()` at inference. `predict()` already does this.
- [ ] Scaler and encoders fitted on train only, then applied to test with the **same** column order.
- [ ] Test data cleaned with the **same** function as train (units!).
- [ ] Time series not shuffled.
- [ ] Checked the threshold on **every** file, not just one.
- [ ] Class labels mapped back from 0..K-1 to the original names.

## Process
- [ ] Ran the sample check first, because wrong submissions likely cost penalty minutes.
- [ ] Uploaded the source code after passing (likely a bonus).
- [ ] Committed and pushed so teammates have the latest cleaning code.
- [ ] Didn't spend more than 20 minutes stuck. Ask a teammate, re-read the statement, look for a shortcut.
