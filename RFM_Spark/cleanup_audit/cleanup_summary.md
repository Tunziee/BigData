# Cleanup summary

Project root: `D:\Doan_bigdata\BigData\BigData\RFM_Spark`

## Outcome

Cleanup completed safely. The net change was limited to disposable Python/test caches. Research data, source, models, predictions, reports, dashboard files, audit evidence and submission artifacts were retained.

| Measure | Before | After | Net change |
|---|---:|---:|---:|
| Files | 4,705 | 3,389 | -1,316 |
| Bytes | 6,678,427,526 | 6,650,632,595 | -27,794,931 |
| Approximate size | 6.22 GiB | 6.19 GiB | -26.51 MiB |

The cleanup log records 151 cache-directory deletion events across the initial cleanup and the follow-up passes required after tests recreated runtime caches. The final inventory contains no cache candidates.

## What was removed

Only these reproducible cache categories were eligible:

- `__pycache__/`
- `.pytest_cache/`
- `.mypy_cache/`
- `.ruff_cache/`
- loose `*.pyc` and `*.pyo`

Compiled dependency binaries such as `.pyd` were explicitly excluded.

## What was protected

The following areas were kept intact: `data/`, `output/`, `reports/`, `deliverables/`, `src/`, `sql/`, `dashboard/`, `.agents/`, `.venv/`, Spark/Hadoop integrity files, serialized models, prediction outputs, report figures, and audit summaries.

The protected manifest check passed: 13 entries were checked, all available hashes and sizes matched, and no protected path was missing.

## Review-only findings

The audit found 176 exact-hash duplicate groups. They were not deleted because they include source snapshots, report/submission copies, selected figures, audit copies, Hadoop CRC evidence and historical physical-plan artifacts. Any future deletion requires a separate manual packaging decision.

The existing `.venv` remains present but its entry point is not portable on this machine because `pyvenv.cfg` references an unavailable Python 3.11 installation. Validation therefore used the installed Python 3.12.10 interpreter; this did not change project artifacts.

## Verification

- Dashboard tests: `6 passed in 3.83s`.
- AST parse: 47 Python files passed.
- Final cache scan: zero cache directories and zero loose bytecode files.
- Browser smoke test: all four dashboard screens passed.
- Git state: pre-existing deletions under `web_dashboard/` and pre-existing untracked dashboard/audit files were preserved; no reset, checkout, clean, commit or push was executed.

## Audit files

- `inventory_before.csv` and `inventory_after.csv` provide before/after file inventories.
- `cleanup_inventory.csv` preserves the full baseline inventory with classifications.
- `dependency_map.md` and `dependency_map.json` record the dependency/usage review.
- `protected_artifacts_manifest.json` records protected paths and checksums.
- `safe_cleanup_deleted.csv` records every cache-directory deletion event.
- `deletion_candidates_review.md` records duplicate groups that remain for manual review.

