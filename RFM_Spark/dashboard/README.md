# RFM Spark — analytical dashboard

This Streamlit layer is a read-only visualization supplement for the locked RFM Spark artifacts. It does not retrain K-Means, rerun FP-Growth, execute Spark SQL, or mutate the locked research pipeline outputs.

## Run

From the project dashboard directory:

```powershell
cd "D:\Doan_bigdata\BigData\BigData\RFM_Spark\dashboard"
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The application expects the `dashboard` directory to be directly below the project root, so that `services/artifacts.py` resolves `D:\Doan_bigdata\BigData\BigData\RFM_Spark\output` without an environment variable.

## Pages

- **Model Comparison** — B0/M1/M2 quality in the selected RFM3D or eRFM5D space.
- **Customer Segments** — locked segment profiles and distributions for M2, M1, or B0.
- **Customer Explorer** — real customer lookup from the eRFM feature dataset and prediction Parquet artifacts.
- **Spark Performance** — controlled runtime, storage, cache, Spark execution evidence, Spark SQL Analytical Layer snapshot, and FP-Growth V2 snapshot.

## Validation

From this directory:

```powershell
python -m pytest tests
```

`dashboard_artifact_inventory.md` documents the authoritative file, schema, function, and limitation for each dashboard data source. The dashboard intentionally does not load the raw transaction table at runtime.
