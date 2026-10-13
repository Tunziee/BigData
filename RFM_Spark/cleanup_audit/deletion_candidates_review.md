# Deletion candidates requiring review

## Automatically removed safe caches

The only automatic deletion scope is disposable Python/test cache:

- `__pycache__/`
- `.pytest_cache/`
- `.mypy_cache/`
- `.ruff_cache/`
- loose `*.pyc` and `*.pyo`

These files are reproducible and are not research artifacts. No `.pyd`, model, dataset, report, source, dashboard or submission file is included.

## Review-required duplicates — not deleted

The audit found 176 exact-hash duplicate groups among hashable files. They remain untouched because duplicates include:

- source snapshots under `deliverables/BigData_Nhom11_Final/02_Source/`;
- selected report copies and final submission copies;
- audit JSON copies under `deliverables/` and final output;
- figure copies needed by the report package;
- Spark `_SUCCESS.crc` files in different output directories;
- historical association-rule physical-plan copies.

Examples:

| Files | Risk | Recommendation |
|---|---|---|
| `deliverables/BigData_Nhom11_Final_Report.docx` and `deliverables/BigData_Nhom11_Final/01_Report/BigData_Nhom11_Final_Report.docx` | HIGH | KEEP both until submission packaging is explicitly revised |
| `src/` scripts and `deliverables/BigData_Nhom11_Final/02_Source/src/` copies | HIGH | KEEP both; one is working source, one is submission snapshot |
| `output/visualization/.../figures/` and selected report figures | MEDIUM | KEEP both; report package depends on the selected copies |
| output `_SUCCESS.crc` files | HIGH | KEEP; they are Hadoop integrity evidence |

No confirmation is requested for the cache cleanup. Confirmation is required before deleting any row in this document.
