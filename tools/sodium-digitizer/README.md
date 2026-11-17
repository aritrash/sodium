# Sodium Digitizer 0.4.0

Reproducible image-derived tube-characteristic data pipeline. LinePilot performs digitization; Sodium Digitizer records provenance, supports explicit manual point exclusion, manual validation, and PCHIP resampling.

## Commands

- `inventory`: inventory source images and SHA-256 hashes.
- `classify`: deterministic filename classification; does not calibrate graphs.
- `run`: run LinePilot using a graph profile. Extraction is never automatically accepted.
- `clean`: manually exclude selected original CSV row indices and/or inclusive X intervals. Writes a cleaned derivative and an audit JSON. No automatic outlier rejection or smoothing.
- `validate`: create a review candidate and sidecar. Without `--accept`, status is `REVIEW_REQUIRED`; with `--accept`, records the operator's explicit manual validation assertion.
- `pchip`: interpolate only candidates with a `MANUALLY_VALIDATED` sidecar. No extrapolation.

## Cleaning

Point indices are zero-based CSV data-row indices (header excluded), referring to the exact input file before any filtering. You can remove individual indices or inclusive index ranges:

```bash
sodium-digitizer clean --input curve.csv --output cleaned/ --drop-indexes 4,8-11
```

Or remove all points in inclusive X intervals:

```bash
sodium-digitizer clean --input curve.csv --output cleaned/ --drop-x-ranges 180:200,250:270
```

Both selectors can be combined. The original CSV is never modified. The audit records the source SHA-256, all removed original row indices, matched X intervals, and point counts. The tool deliberately does not infer outliers from smoothness or curvature.

## Full workflow

```bash
sodium-digitizer clean --input digitized/raw/curve.csv --output digitized/candidate/
sodium-digitizer validate --input digitized/candidate/curve_cleaned.csv --output validated/curves/ --x-column x --y-column y
```

Inspect the curve and validation candidate. After reviewing trace identity, calibration, continuity, and physical plausibility, explicitly record acceptance:

```bash
sodium-digitizer validate --input digitized/candidate/curve_cleaned.csv --output validated/curves/ --x-column x --y-column y --accept --reviewer aritrash
```

Then interpolate:

```bash
sodium-digitizer pchip --input validated/curves/ --output interpolated/pchip_1v/ --step 1.0 --x-column x --y-column y
```

PCHIP output is a derived estimate, not a new physical measurement. It carries `PCHIP_INTERPOLATED`, source filename and source hash.

## Scientific safeguards

- Keep original LinePilot output immutable.
- Manual point exclusion only; no automatic outlier removal or smoothing.
- Never interpolate raw or unvalidated extraction output.
- Never extrapolate beyond the measured X domain.
- Resolve duplicate X values explicitly before validation.
- Keep empirical, interpolated, and synthetic data separate.
- Split evaluation by complete curves or operating regions, not random neighbouring samples.
