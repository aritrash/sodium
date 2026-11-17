# RCA 12AX7A calibration profiles — 0.1.0

This is an initial profile inventory derived from the 25 source JPEGs.

- Filename-derived labels and scale hints are provisional until checked against the rendered axes.
- Only `ik_vs_vp_primary.json` contains a first-pass ROI and tick calibration.
- The primary profile is explicitly `PROVISIONAL`; do not use it as validated empirical ground truth.
- Preserve `Ik` as cathode current.
- The 3D plots and TrioPar5 main panel are reference-only in the 2D extraction workflow.
- Curve colours and grid-voltage assignments require manual verification, especially where traces overlap or appear doubled.

Next review: verify ROI edges and each tick's pixel centre against the original full-resolution image, then verify curve identities from the legends.
