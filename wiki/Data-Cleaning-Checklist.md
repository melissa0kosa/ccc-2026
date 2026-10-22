# Data Cleaning Checklist

CCC data is deliberately dirty. Past rounds included temperatures spelled out in English, °F mixed with °C, Kelvin, "?" units, NaNs and out-of-range rows.

```python
df = read_csv(path)
df.info(); df.describe(include="all").T; df.isna().sum(); df.nunique()
df.sample(10)
```

- [ ] **Dtypes:** are numeric columns `object`? Find the non-numeric values: `df[c][pd.to_numeric(df[c], errors="coerce").isna()].unique()`.
- [ ] **Spelled-out numbers** ("twenty-three"): map them with a small dict or a parser, then verify.
- [ ] **Units mixed in one column:** look at the histogram. Bimodal usually means two units. Convert, for example °F to °C: `(f-32)*5/9`, K to °C: `k-273.15`.
- [ ] **Units given per row** ("12C", "54F"): split the number from the suffix.
- [ ] **Unknown unit "?":** infer it from the min/max range.
- [ ] **Decimal commas** or thousand separators.
- [ ] **NaN and sentinel values** (-1, 999, 0 where impossible). Decide: drop, impute (median), or add an "is_missing" flag.
- [ ] **Out-of-range values** (negative humidity, > 100%): drop them or clip them.
- [ ] **Duplicates:** check exact duplicates and duplicate ids with different labels.
- [ ] **Train/test overlap:** compare the train and test distributions with `describe()` side by side.
- [ ] **Categoricals:** use one-hot for low cardinality and target/ordinal encoding (or CatBoost) for high cardinality.
- [ ] **Dates:** parse them, then extract dayofweek, month, dayofyear and trend index. Sort by time.
- [ ] **Entity structure:** do rows belong to groups (flock, user, location)? Aggregate per group or use GroupKFold.
- [ ] **Target:** look at its distribution. If it is skewed, train on `log1p(y)` and invert with `expm1`.

Put every cleaning step in **one function** in `levels/common.py` or similar. Later levels reuse the same data.
