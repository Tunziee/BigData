# Post-cleanup regression results

Project root: `D:\Doan_bigdata\BigData\BigData\RFM_Spark`

## Scope

The cleanup removed only disposable Python/test caches. No source code, data, model, prediction, report, dashboard, or submission artifact was regenerated or changed.

## Automated checks

| Check | Result | Evidence |
|---|---|---|
| Existing project `.venv` interpreter | BLOCKED / pre-existing | `pyvenv.cfg` points to the unavailable `C:\Users\VuongNguyen\AppData\Local\Programs\Python\Python311\python.exe` |
| Dashboard pytest | PASS | `6 passed in 3.83s` using Python 3.12.10 |
| Python AST parse | PASS | 47 files under `src/` and `dashboard/` parsed successfully |
| Protected artifact manifest | PASS | 13 entries checked; 9 SHA-256 hashes matched; 2 large files retained and size-checked; no mismatches |
| Final cache scan | PASS | 0 cache directories and 0 loose `.pyc`/`.pyo` files |

## Dashboard smoke test after cleanup

The dashboard was launched locally with Python 3.12.10 and checked through browser automation. All four screens passed:

- Model Comparison: PASS; B0/M1/M2 table, RFM3D/eRFM5D selector and comparison figures loaded.
- Customer Segments: PASS; M2 / Enhanced K-Means profile, five clusters, distribution, heatmap, radar and profile table loaded.
- Customer Explorer: PASS; a current real customer identifier from the eRFM artifact loaded overview values, B0/M1/M2 predictions, feature comparison and explanation.
- Spark Performance: PASS; Model runtime, CSV vs Parquet, Cache and Spark execution tabs loaded with their locked values.

The direct-route browser harness can briefly show a Streamlit page-not-found overlay while the app initializes; normal sidebar/direct navigation works after initialization. This behavior was observed before and after cleanup and is not caused by the cleanup.

## Protected values observed after cleanup

- M2 remains the Enhanced K-Means model with K=5.
- Spark SQL and FP-Growth V2 read-only summaries remain available.
- Runtime, storage/query, cache and Spark execution evidence remained readable.
- The final comparison and visualization artifacts remained byte/hash consistent with the protected manifest.
