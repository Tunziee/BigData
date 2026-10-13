# Baseline test results before cleanup

Project root: `D:\Doan_bigdata\BigData\BigData\RFM_Spark`

## Repository state captured

- Git repository root: `D:\Doan_bigdata\BigData\BigData`
- Branch: `main`
- Existing Git changes were preserved as-is: deleted `web_dashboard/`, untracked `.agents/`, untracked `dashboard/` and untracked `dashboard_artifact_inventory.md`.
- No reset, checkout, clean, commit or push was executed.

## Inventory baseline

- Files scanned: 4,705.
- Total size: 6,678,427,526 bytes, approximately 6.22 GiB.
- Protected research/source areas: `data/`, `output/`, `reports/`, `deliverables/`, `src/`, `sql/`, `dashboard/`, `.agents/` and `.venv/`.
- Cache candidates identified conservatively: 1,316 files in 143 cache directories, 27,794,931 bytes.
- `.pyd` dependency binaries were explicitly excluded from deletion candidates.
- 176 exact-duplicate groups were found among files small enough to hash. Duplicate groups were not deleted because many are source/report/submission snapshots.

## Python checks

| Test | Result | Evidence |
|---|---|---|
| Existing `.venv` test command | BLOCKED | `.venv\pyvenv.cfg` points to missing `C:\Users\VuongNguyen\AppData\Local\Programs\Python\Python311\python.exe` |
| Dashboard artifact pytest with current Python 3.12.10 | PASS | `6 passed in 4.66s` |
| AST parse of `src/` and `dashboard/` Python files | PASS | 47 files parsed |

## Dashboard baseline

Started with Python 3.12.10 on localhost and checked through browser automation:

- Model Comparison: PASS; B0/M1/M2 table loaded, RFM3D/eRFM5D control present.
- Customer Segments: PASS; M2 profile loaded with 5 clusters and 1,362,281 customers.
- Customer Explorer: PASS; a real customer ID loaded feature cards and B0/M1/M2 predictions.
- Spark Performance: PASS; model runtime, CSV vs Parquet, Cache and Spark execution tabs loaded.
- Spark SQL Analytical Layer and FP-Growth Association-Rule Mining V2 snapshot loaded.

Known baseline note: opening a deep route directly in the browser harness briefly showed a Streamlit “Page not found” overlay; closing it and navigating through the sidebar worked normally. This was recorded as pre-existing harness behavior, not attributed to cleanup.

## Protected baseline values

- B0/M1/M2 quality matrix, M2 K=5, cohort and comparison metrics.
- Runtime benchmark, storage/query benchmark, cache benchmark and Spark execution evidence.
- Spark SQL summary and FP-Growth V2 summary.
- Full raw/cleaned data, models, predictions, profiles, reports and submission artifacts.

The machine-readable snapshots are in `cleanup_audit/protected_artifacts_manifest.json` and `cleanup_audit/inventory_before.csv`.
