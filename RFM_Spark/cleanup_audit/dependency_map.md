# Dependency and duplicate audit

This map is conservative: lack of an import match is not proof that a file is unused.

- Python source files: 1245
- Dashboard Python files: 15
- Exact duplicate groups among hashed files: 176
- Large-file hashing limit: 67108864 bytes

## Protected areas

`data/`, `output/`, `reports/`, `deliverables/`, `src/`, `sql/`, `dashboard/`, `.agents/` and `.venv/` are protected from automatic cleanup.

## Dependency rule

The audit does not delete or move source, model, report, dataset, manifest, checksum, or dashboard files.
