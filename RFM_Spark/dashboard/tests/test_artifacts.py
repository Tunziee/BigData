from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.artifacts import (  # noqa: E402
    load_cache,
    load_customer_lookup,
    load_execution,
    load_fp_growth,
    load_model_quality,
    load_runtime,
    load_shuffle_metrics,
    load_stage27_summary,
    load_sql,
    load_storage,
    sample_customer_id,
)


def test_locked_model_quality_readback() -> None:
    quality = load_model_quality()
    assert set(quality["ModelID"]) == {"B0", "M1", "M2"}
    assert quality.loc[quality["ModelID"].eq("M2"), "K"].iloc[0] == 5


def test_locked_m2_sql_segment_distribution() -> None:
    *_, segments = load_sql()
    assert len(segments) == 5
    assert int(segments["customer_count"].sum()) == 1_362_281


def test_locked_benchmark_readback() -> None:
    runtime = load_runtime()
    assert runtime.loc[runtime["ModelID"].eq("M2"), "MedianTotalFitPlusAssignSec"].iloc[0] < 14
    storage, storage_runtime = load_storage()
    assert set(storage["format"]) == {"CSV", "Parquet"}
    assert "CUSTOMER_AGGREGATION" in set(storage_runtime["QueryName"])
    cache_df, cache_summary = load_cache()
    assert "Q3_CUSTOMER_AGGREGATION" in set(cache_df["QueryID"])
    assert cache_summary["queries"]["Q3"]["break_even_executions"] == 39


def test_locked_execution_evidence() -> None:
    execution = load_execution()
    assert execution["execution_evidence_complete"] is True
    assert execution["q2"]["stage_count"] == 3
    assert execution["q3"]["shuffle_read_bytes"] == 327_776_023
    shuffle = load_shuffle_metrics()
    assert int(shuffle.loc[shuffle["Query"].eq("Q3_CUSTOMER_AGGREGATION"), "shuffle_read_records"].iloc[0]) == 4_271_402
    assert load_stage27_summary()["transaction_row_count"] == 31_788_324


def test_locked_fp_growth_readback() -> None:
    thresholds, rules, summary = load_fp_growth()
    selected = thresholds.loc[thresholds["selected"].eq(True)]
    assert float(selected["min_support"].iloc[0]) == 0.0002
    assert len(rules) == 104
    assert summary["association_rule_count"] == 104
    assert summary["manual_rule_validation_pass"] is True


def test_real_customer_lookup() -> None:
    customer_id = sample_customer_id()
    assert customer_id is not None
    lookup = load_customer_lookup(customer_id)
    assert lookup["features"] is not None
    assert lookup["features"]["customer_id"] is not None
    assert set(lookup["predictions"]) == {"B0", "M1", "M2"}
    assert load_customer_lookup("not-a-real-customer-id")["features"] is None
