"""Stage 29 - build the final report-ready visualization layer from locked artifacts.

This script is intentionally visualization-only. It does not create a Spark session,
fit a model, recompute a benchmark, or modify any Stage 1-28 artifact.
"""
from __future__ import annotations

import html
import json
import math
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

# The project keeps its scientific dependencies in the project virtual environment.
for candidate in (
    Path(r"D:/Doan_bigdata/BigData/BigData/RFM_Spark/.venv/Lib/site-packages"),
    Path(r"C:/Users/tayb1/AppData/Local/Programs/Python/Python312/Lib/site-packages"),
):
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from PIL import Image, ImageDraw, ImageFont


BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "output" / "visualization" / "stage29"
FIG_DIR = OUT / "figures"
DATA_DIR = OUT / "data"
MANIFEST_DIR = OUT / "manifests"
REPORT = BASE / "reports" / "stage29_final_visualization_layer.txt"
RUN_LOG = BASE / "reports" / "stage29_run.log"

AUTHORITY_STAGE27 = "SPARK_SQL_ANALYTICAL_LAYER_V1"
AUTHORITY_STAGE28 = "FP_GROWTH_ASSOCIATION_RULES_V2"
AUTHORITY_STAGE29 = "FINAL_VISUALIZATION_LAYER_V1"

FONT_REG = Path(r"C:/Windows/Fonts/segoeui.ttf")
FONT_BOLD = Path(r"C:/Windows/Fonts/segoeuib.ttf")

COLORS = ["#275DAD", "#D97706", "#168A6A", "#8B5CF6", "#C2415A", "#596273"]
INK = "#172033"
GRID = "#D7DEE9"
MUTED = "#5B6575"
SURFACE = "#FBFCFE"


def load_json(relative: str):
    p = BASE / relative
    if not p.exists():
        raise FileNotFoundError(f"Required authority artifact missing: {p}")
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_csv(relative: str, dtype=None) -> pd.DataFrame:
    p = BASE / relative
    if not p.exists():
        raise FileNotFoundError(f"Required source CSV missing: {p}")
    return pd.read_csv(p, dtype=dtype)


def load_parquet(relative: str) -> pd.DataFrame:
    p = BASE / relative
    if not p.exists():
        raise FileNotFoundError(f"Required source Parquet missing: {p}")
    return ds.dataset(str(p), format="parquet").to_table().to_pandas()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(value, fh, ensure_ascii=False, indent=2, default=json_default)


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def scalar(value):
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return value


def font(size: int, bold: bool = False):
    path = FONT_BOLD if bold and FONT_BOLD.exists() else FONT_REG
    try:
        return ImageFont.truetype(str(path), size)
    except Exception:
        return ImageFont.load_default()


def esc(text: str) -> str:
    return html.escape(str(text), quote=True)


class Scene:
    """Small deterministic scene graph rendered identically to PNG and SVG."""

    def __init__(self, title: str, subtitle: str = "", width: int = 1400, height: int = 850):
        self.width, self.height = width, height
        self.title, self.subtitle = title, subtitle
        self.primitives = []

    def text(self, x, y, value, size=20, fill=INK, bold=False, anchor="start", rotate=0):
        self.primitives.append(("text", float(x), float(y), str(value), int(size), fill, bool(bold), anchor, rotate))

    def line(self, x1, y1, x2, y2, fill=GRID, width=2, dash=False):
        self.primitives.append(("line", float(x1), float(y1), float(x2), float(y2), fill, int(width), bool(dash)))

    def rect(self, x, y, w, h, fill, outline=None, width=1):
        self.primitives.append(("rect", float(x), float(y), float(w), float(h), fill, outline, int(width)))

    def circle(self, x, y, r, fill, outline=None, width=1):
        self.primitives.append(("circle", float(x), float(y), float(r), fill, outline, int(width)))

    def polyline(self, points, fill, width=3, closed=False):
        self.primitives.append(("polyline", [(float(x), float(y)) for x, y in points], fill, int(width), bool(closed)))

    def finish(self):
        self.text(70, 48, self.title, size=32, bold=True)
        if self.subtitle:
            self.text(72, 83, self.subtitle, size=17, fill=MUTED)

    def save(self, stem: str):
        self.finish()
        img = Image.new("RGB", (self.width, self.height), SURFACE)
        draw = ImageDraw.Draw(img)
        for primitive in self.primitives:
            kind = primitive[0]
            if kind == "text":
                _, x, y, value, size, fill, bold, anchor, rotate = primitive
                f = font(size, bold)
                if rotate:
                    layer = Image.new("RGBA", (600, 100), (0, 0, 0, 0))
                    ld = ImageDraw.Draw(layer)
                    ld.text((0, 0), value, font=f, fill=fill)
                    layer = layer.rotate(rotate, expand=True)
                    img.paste(layer, (int(x), int(y)), layer)
                else:
                    bbox = draw.textbbox((0, 0), value, font=f)
                    tw = bbox[2] - bbox[0]
                    xx = x - tw if anchor == "end" else x - tw / 2 if anchor == "middle" else x
                    draw.text((int(xx), int(y)), value, font=f, fill=fill)
            elif kind == "line":
                _, x1, y1, x2, y2, fill, width, dash = primitive
                if dash:
                    total = max(1, int(math.hypot(x2 - x1, y2 - y1)))
                    for i in range(0, total, 14):
                        j = min(total, i + 7)
                        xa, ya = x1 + (x2 - x1) * i / total, y1 + (y2 - y1) * i / total
                        xb, yb = x1 + (x2 - x1) * j / total, y1 + (y2 - y1) * j / total
                        draw.line((xa, ya, xb, yb), fill=fill, width=width)
                else:
                    draw.line((x1, y1, x2, y2), fill=fill, width=width)
            elif kind == "rect":
                _, x, y, w, h, fill, outline, width = primitive
                draw.rectangle((x, y, x + w, y + h), fill=fill, outline=outline, width=width)
            elif kind == "circle":
                _, x, y, r, fill, outline, width = primitive
                draw.ellipse((x - r, y - r, x + r, y + r), fill=fill, outline=outline, width=width)
            elif kind == "polyline":
                _, points, fill, width, closed = primitive
                pts = points + ([points[0]] if closed and points else [])
                draw.line(pts, fill=fill, width=width, joint="curve")
        img.save(FIG_DIR / f"{stem}.png", dpi=(300, 300))
        svg = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" viewBox="0 0 {self.width} {self.height}">',
            f'<rect width="100%" height="100%" fill="{SURFACE}"/>',
        ]
        for primitive in self.primitives:
            kind = primitive[0]
            if kind == "text":
                _, x, y, value, size, fill, bold, anchor, rotate = primitive
                weight = 700 if bold else 400
                transform = f' transform="rotate({rotate} {x} {y})"' if rotate else ""
                svg.append(f'<text x="{x:g}" y="{y:g}" fill="{fill}" font-family="Segoe UI, Arial, sans-serif" font-size="{size}px" font-weight="{weight}" text-anchor="{anchor}"{transform}>{esc(value)}</text>')
            elif kind == "line":
                _, x1, y1, x2, y2, fill, width, dash = primitive
                dash_attr = ' stroke-dasharray="8 8"' if dash else ""
                svg.append(f'<line x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" y2="{y2:g}" stroke="{fill}" stroke-width="{width}"{dash_attr}/>')
            elif kind == "rect":
                _, x, y, w, h, fill, outline, width = primitive
                stroke = f' stroke="{outline}" stroke-width="{width}"' if outline else ""
                svg.append(f'<rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" fill="{fill}"{stroke}/>')
            elif kind == "circle":
                _, x, y, r, fill, outline, width = primitive
                stroke = f' stroke="{outline}" stroke-width="{width}"' if outline else ""
                svg.append(f'<circle cx="{x:g}" cy="{y:g}" r="{r:g}" fill="{fill}"{stroke}/>')
            elif kind == "polyline":
                _, points, fill, width, closed = primitive
                p = " ".join(f"{x:g},{y:g}" for x, y in points + ([points[0]] if closed and points else []))
                tag = "polygon" if closed else "polyline"
                svg.append(f'<{tag} points="{p}" fill="none" stroke="{fill}" stroke-width="{width}" stroke-linejoin="round"/>')
        svg.append("</svg>")
        (FIG_DIR / f"{stem}.svg").write_text("\n".join(svg), encoding="utf-8")


def write_source(stem: str, frame: pd.DataFrame) -> str:
    frame = frame.copy()
    frame.to_csv(DATA_DIR / f"{stem}.csv", index=False, encoding="utf-8-sig")
    return f"data/{stem}.csv"


def finite_frame(frame: pd.DataFrame, label: str):
    for col in frame.columns:
        if pd.api.types.is_numeric_dtype(frame[col]):
            values = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite plotted value in {label}.{col}")


def axis(scene, x0, y0, x1, y1, xmin, xmax, ymin, ymax, x_ticks, y_ticks, xlabel="", ylabel=""):
    def sx(v): return x0 + (float(v) - xmin) / (xmax - xmin or 1) * (x1 - x0)
    def sy(v): return y1 - (float(v) - ymin) / (ymax - ymin or 1) * (y1 - y0)
    for tick in y_ticks:
        v, label = tick if isinstance(tick, (tuple, list)) else (tick, f"{tick:g}")
        yy = sy(v)
        scene.line(x0, yy, x1, yy, GRID, 1)
        scene.text(x0 - 16, yy - 11, str(label), size=15, fill=MUTED, anchor="end")
    scene.line(x0, y1, x1, y1, INK, 2)
    scene.line(x0, y0, x0, y1, INK, 2)
    for v, label in x_ticks:
        xx = sx(v)
        scene.line(xx, y1, xx, y1 + 8, INK, 2)
        scene.text(xx, y1 + 15, label, size=15, fill=MUTED, anchor="middle")
    if xlabel:
        scene.text((x0 + x1) / 2, y1 + 55, xlabel, size=17, fill=INK, anchor="middle")
    if ylabel:
        scene.text(x0 - 84, (y0 + y1) / 2, ylabel, size=17, fill=INK, anchor="middle", rotate=90)
    return sx, sy


def bar_chart(stem, title, subtitle, labels, values, ylabel, colors=None, value_fmt="{:.2f}"):
    scene = Scene(title, subtitle)
    x0, y0, x1, y1 = 150, 145, 1330, 690
    ymax = max(values) * 1.2 if values and max(values) > 0 else 1
    ticks = np.linspace(0, ymax, 5)
    sx, sy = axis(scene, x0, y0, x1, y1, 0, len(labels), 0, ymax, [(i + 0.5, str(l)) for i, l in enumerate(labels)], [(float(v), f"{v:g}") for v in ticks], ylabel=ylabel)
    bw = (x1 - x0) / max(1, len(labels)) * 0.65
    for i, value in enumerate(values):
        xx = sx(i + 0.5) - bw / 2
        yy = sy(value)
        scene.rect(xx, yy, bw, y1 - yy, (colors or COLORS)[i % len(colors or COLORS)])
        scene.text(xx + bw / 2, yy - 28, value_fmt.format(value), size=15, fill=INK, anchor="middle")
    scene.save(stem)


def line_chart(stem, title, subtitle, x_values, series, xlabel, ylabel, xlabels=None):
    scene = Scene(title, subtitle)
    x0, y0, x1, y1 = 150, 145, 1330, 690
    vals = [v for _, ys in series for v in ys]
    ymin, ymax = min(vals), max(vals)
    pad = (ymax - ymin) * 0.12 or 1
    ymin, ymax = ymin - pad, ymax + pad
    if xlabels is None: xlabels = [str(x) for x in x_values]
    ticks = [(i, xlabels[i]) for i in range(len(x_values))]
    sx, sy = axis(scene, x0, y0, x1, y1, min(x_values), max(x_values), ymin, ymax, ticks, [(float(v), f"{v:.2f}") for v in np.linspace(ymin, ymax, 5)], xlabel=xlabel, ylabel=ylabel)
    for idx, (name, ys) in enumerate(series):
        points = [(sx(x), sy(v)) for x, v in zip(x_values, ys)]
        scene.polyline(points, COLORS[idx % len(COLORS)], width=4)
        for x, y in points:
            scene.circle(x, y, 6, COLORS[idx % len(COLORS)])
        scene.text(x1 - 10, y0 + 25 + idx * 28, name, size=16, fill=COLORS[idx % len(COLORS)], anchor="end")
    scene.save(stem)


def get_feature_rows(summary, dataset=None):
    rows = summary.get("features", [])
    if isinstance(rows, str):
        rows = json.loads(rows)
    if dataset:
        rows = [r for r in rows if str(r.get("Dataset", "")).lower() == dataset.lower()]
    return rows


def main():
    clean_namespace = not OUT.exists() and not REPORT.exists() and not RUN_LOG.exists()
    if not clean_namespace:
        raise RuntimeError("Stage 29 clean namespace gate failed; remove only the Stage 29 namespace/report/run-log before rerun.")

    OUT.mkdir(parents=True, exist_ok=False)
    FIG_DIR.mkdir(); DATA_DIR.mkdir(); MANIFEST_DIR.mkdir(); REPORT.parent.mkdir(parents=True, exist_ok=True)

    # Read-only authority gates.
    s12 = load_json("output/analysis/stage12/transformation_summary.json")
    s12_skew = load_json("output/analysis/stage12/skewness_comparison.json")
    s9 = load_json("output/analysis/stage9/optimal_k_selection.json")
    s13 = load_json("output/analysis/stage13/enhanced_kmeans_k_sweep.json")
    s16 = load_json("output/analysis/stage16/final_enhanced_cluster_profile_summary.json")
    s22 = load_json("output/comparison/stage22/stage22_final_summary.json")
    s23 = load_json("output/benchmark/stage23/stage23_final_summary.json")
    s24 = load_json("output/benchmark/stage24/stage24_final_summary.json")
    s25 = load_json("output/benchmark/stage25/stage25_final_summary.json")
    s26 = load_json("output/analysis/stage26/stage26_final_summary.json")
    s27 = load_json("output/sql/stage27/stage27_final_summary.json")
    s28 = load_json("output/fpgrowth/stage28_v2/stage28_v2_final_summary.json")
    s28_v1 = load_json("output/fpgrowth/stage28/stage28_final_summary.json")
    s28_v1_config = load_json("output/fpgrowth/stage28/fp_growth_config.json")
    s28_v1_items = load_json("output/fpgrowth/stage28/frequent_itemset_summary.json")
    enriched_rules = load_json("output/fpgrowth/stage28_v2/enriched_top_rules.json")

    expected_authorities = {
        "stage12": (s12, "IMPROVED_ERFM_FEATURE_SPACE"),
        "stage13": (s13, "ENHANCED_KMEANS_K_SWEEP"),
        "stage16": (s16, "ENHANCED_KMEANS_FINAL_CLUSTER_PROFILE"),
        "stage22": (s22, "PRIMARY_B0_M1_M2_QUALITY_COMPARISON_V1"),
        "stage23": (s23, "PRIMARY_B0_M1_M2_RUNTIME_BENCHMARK_1M_V1"),
        "stage24": (s24, "CSV_PARQUET_IO_BENCHMARK_V1"),
        "stage25": (s25, "CACHE_NO_CACHE_BENCHMARK_V1"),
        "stage26": (s26, "SPARK_EXECUTION_EVIDENCE_V1"),
        "stage27": (s27, AUTHORITY_STAGE27),
        "stage28_v2": (s28, AUTHORITY_STAGE28),
        "stage28_v1": (s28_v1, "FP_GROWTH_ASSOCIATION_RULES_V1"),
    }
    for name, (obj, authority) in expected_authorities.items():
        if obj.get("authority") != authority:
            raise ValueError(f"Authority gate failed for {name}: {obj.get('authority')} != {authority}")
    if s28.get("ready_for_stage29") is not True or s28.get("fp_growth_analysis_complete") is not True:
        raise ValueError("Stage 28 V2 is not ready/complete")
    if s28.get("models_modified") or s28.get("stage27_modified") or s28.get("v1_modified"):
        raise ValueError("Stage 28 V2 reports an upstream modification")
    if s22.get("models_retrained") or s22.get("gmm_executed"):
        raise ValueError("Stage 22 reports an invalid comparison path")
    for obj in (s23, s24, s25, s26, s27):
        if obj.get("models_modified") or obj.get("primary_runtime_authority_modified") or obj.get("stage24_modified") or obj.get("stage25_modified") or obj.get("stage23_modified") or obj.get("stage26_modified"):
            raise ValueError("Locked benchmark/SQL authority indicates an upstream modification")

    rfm_summary = load_json("output/analysis/rfm_distribution/summary.json")
    erfm_summary = load_json("output/analysis/erfm_distribution/summary.json")
    stage8 = load_csv("output/analysis/stage8/kmeans_k_sweep.csv")
    stage13 = load_csv("output/analysis/stage13/enhanced_kmeans_k_sweep.csv")
    stage22_matrix = load_csv("output/comparison/stage22/primary_three_model_comparison_matrix.csv")
    stage23_a = load_csv("output/benchmark/stage23/track_a_assignment_summary.csv")
    stage23_b = load_csv("output/benchmark/stage23/track_b_build_assign_summary.csv")
    stage24_runtime = load_csv("output/benchmark/stage24/csv_vs_parquet_benchmark_summary.csv")
    stage25_runtime = load_csv("output/benchmark/stage25/cache_vs_no_cache_summary.csv")
    q2 = load_csv("output/sql/stage27/q2_monthly_purchase_trend.csv")
    q1_kpi = load_json("output/sql/stage27/q1_dataset_kpi.json")
    q4 = load_csv("output/sql/stage27/q4_top20_product_groups.csv")
    q5 = load_csv("output/sql/stage27/q5_age_band_analysis.csv")
    q6 = load_csv("output/sql/stage27/q6_m2_segment_analytics.csv")
    threshold = load_csv("output/fpgrowth/stage28_v2/threshold_sensitivity.csv")
    rules_lift = load_csv("output/fpgrowth/stage28_v2/top20_rules_by_lift.csv", dtype={"antecedent": str, "consequent": str})
    rules_conf = load_csv("output/fpgrowth/stage28_v2/top20_rules_by_confidence.csv", dtype={"antecedent": str, "consequent": str})
    rules_all = load_parquet("output/fpgrowth/stage28_v2/selected_association_rules.parquet")
    sse = load_json("output/comparison/stage22/common_space_sse.json")
    silhouettes = load_json("output/comparison/stage22/common_space_silhouette.json")
    balance = load_json("output/comparison/stage22/cluster_balance_metrics.json")
    agreement = load_json("output/comparison/stage22/b0_m1_label_alignment.json")
    shuffle = load_json("output/analysis/stage26/shuffle_metrics.json")

    manifests = []
    sources = {}

    # 01: quantile distribution panels from the locked distribution summaries.
    dist_rows = []
    for dataset, summary in (("RFM", rfm_summary), ("eRFM", erfm_summary)):
        for row in get_feature_rows(summary):
            dist_rows.append({"dataset": dataset, "feature": row["Feature"], **{k.lower(): row.get(k) for k in ("Min", "P01", "P05", "P25", "P50", "P75", "P95", "P99", "Max", "Skewness")}})
    dist = pd.DataFrame(dist_rows)
    finite_frame(dist, "fig01")
    stem = "fig01_rfm_erfm_distributions"; sources[stem] = write_source(stem, dist)
    scene = Scene("RFM and eRFM feature distributions", "Locked quantile profiles; P25-P75 is the central band and P05-P95 is the broad band")
    features = ["Recency", "Frequency", "Monetary", "AvgBasketSize", "ProductVariety"]
    panels = [("RFM", "#275DAD"), ("eRFM", "#168A6A")]
    for pi, (dataset, color) in enumerate(panels):
        rows = [r for r in dist_rows if r["dataset"] == dataset]
        for fi, feat in enumerate(features):
            row = next((r for r in rows if r["feature"] == feat), None)
            if row is None: continue
            col, rr = fi % 3, fi // 3 + pi * 2
            x, y, w, h = 80 + col * 430, 150 + rr * 270, 350, 170
            scene.text(x, y - 25, f"{dataset} · {feat}", size=19, bold=True)
            lo, hi = float(row["p05"]), float(row["p95"]); med = float(row["p50"])
            span = max(hi - lo, 1e-9)
            def px(v): return x + (float(v) - lo) / span * w
            scene.line(x, y + h / 2, x + w, y + h / 2, GRID, 4)
            scene.rect(px(row["p25"]), y + h / 2 - 25, max(3, px(row["p75"]) - px(row["p25"])), 50, color)
            scene.line(px(med), y + 30, px(med), y + h - 30, INK, 4)
            scene.text(x, y + h + 12, f"P05 {lo:g}", size=14, fill=MUTED)
            scene.text(x + w, y + h + 12, f"P95 {hi:g}", size=14, fill=MUTED, anchor="end")
    scene.save(stem)
    manifests.extend([(stem, "RFM/eRFM quantile profiles", ["Stage 5/6 distribution summaries", "Stage 12"] )])

    # 02: skewness transformation comparison.
    skew_rows = []
    for row in get_feature_rows(erfm_summary):
        skew_rows.append({"feature": row["Feature"], "raw_skewness": row.get("Skewness"), "log1p_skewness": row.get("Log1pSkewness") if row.get("Log1pSkewness") is not None else row.get("Skewness"), "skewness_reduction_pct": row.get("SkewnessReductionPct") if row.get("SkewnessReductionPct") is not None else 0.0})
    skew_df = pd.DataFrame(skew_rows); finite_frame(skew_df, "fig02")
    stem = "fig02_skewness_transformation"; sources[stem] = write_source(stem, skew_df)
    scene = Scene("Skewness before and after log1p", "Stage 12 transformation authority; Recency remains untransformed by protocol")
    labels = skew_df.feature.tolist(); x0, y0, x1, y1 = 150, 150, 1330, 680
    ymax = float(skew_df[["raw_skewness", "log1p_skewness"]].max().max()) * 1.2
    sx, sy = axis(scene, x0, y0, x1, y1, 0, len(labels), 0, ymax, [(i + .5, l) for i, l in enumerate(labels)], [(float(v), f"{v:.1f}") for v in np.linspace(0, ymax, 5)], ylabel="Skewness")
    bw = (x1 - x0) / len(labels) * .28
    for i, row in skew_df.iterrows():
        for j, col in enumerate(("raw_skewness", "log1p_skewness")):
            value = float(row[col]); xx = sx(i + .5) + (j - .5) * bw - bw / 2; yy = sy(value)
            scene.rect(xx, yy, bw, y1 - yy, COLORS[j])
            scene.text(xx + bw / 2, yy - 25, f"{value:.2f}", size=14, anchor="middle")
    scene.text(x1 - 10, y0 + 20, "Raw", size=16, fill=COLORS[0], anchor="end")
    scene.text(x1 - 10, y0 + 48, "log1p", size=16, fill=COLORS[1], anchor="end")
    scene.save(stem); manifests.append((stem, "Skewness transformation", ["Stage 12"]));

    # 03-04: K sweeps.
    def sweep_pair(df, prefix, title):
        k = df["k"].astype(int).tolist(); sil_col = "silhouette"; wcss_col = "wcss" if "wcss" in df else "wcss_trainingCost"
        for suffix, ylabel, values in (("a", "Silhouette", df[sil_col].astype(float).tolist()), ("b", "WCSS / training cost", df[wcss_col].astype(float).tolist())):
            stem = {"fig03a": "fig03a_main_silhouette_by_k", "fig03b": "fig03b_main_wcss_by_k", "fig04a": "fig04a_enhanced_silhouette_by_k", "fig04b": "fig04b_enhanced_wcss_by_k"}[f"{prefix}{suffix}"]; frame = pd.DataFrame({"k": k, "value": values, "metric": ylabel}); finite_frame(frame, stem); sources[stem] = write_source(stem, frame)
            line_chart(stem, f"{title}: {ylabel}", "Locked k=2..8 sweep; no refit performed in Stage 29", k, [(ylabel, values)], "K", ylabel, [str(v) for v in k])
            manifests.append((stem, f"{title} {ylabel}", ["Stage 8" if prefix == "fig03" else "Stage 13"]))
    sweep_pair(stage8, "fig03", "Main RFM K-Means")
    sweep_pair(stage13, "fig04", "Enhanced eRFM K-Means")

    # 05-07: primary comparison and balance.
    comp = stage22_matrix[["ModelID", "Silhouette_RFM3D", "Silhouette_ERFM5D", "SSEPerCustomer_RFM3D", "SSEPerCustomer_ERFM5D", "MinClusterRatioPct", "MaxClusterRatioPct", "ClusterSizeCV", "NormalizedEntropy"]].copy()
    finite_frame(comp, "fig05")
    stem = "fig05_primary_model_quality_comparison"; sources[stem] = write_source(stem, comp)
    scene = Scene("Primary model quality comparison", "Common 1M cohort; each metric is read from the locked Stage 22 comparison matrix")
    labels = comp.ModelID.tolist(); metrics = [("Silhouette RFM3D", "Silhouette_RFM3D"), ("Silhouette eRFM5D", "Silhouette_ERFM5D"), ("Entropy", "NormalizedEntropy")]
    x0, y0, x1, y1 = 150, 150, 1330, 680; ymax = 1.0
    sx, sy = axis(scene, x0, y0, x1, y1, 0, len(labels), 0, ymax, [(i + .5, l) for i, l in enumerate(labels)], [(v, f"{v:.1f}") for v in np.linspace(0, 1, 6)], ylabel="Score")
    bw = (x1 - x0) / len(labels) * .22
    for i, row in comp.iterrows():
        for j, (_, col) in enumerate(metrics):
            value = float(row[col]); xx = sx(i + .5) + (j - 1) * bw - bw / 2; yy = sy(value); scene.rect(xx, yy, bw, y1 - yy, COLORS[j]); scene.text(xx + bw / 2, yy - 25, f"{value:.2f}", size=13, anchor="middle")
    for j, (name, _) in enumerate(metrics): scene.text(x1 - 10, y0 + 18 + j * 25, name, size=15, fill=COLORS[j], anchor="end")
    scene.save(stem); manifests.append((stem, "B0/M1/M2 quality comparison", ["Stage 22"]))

    sse_rows = []
    for mid in ("B0", "M1", "M2"):
        for space in ("RFM3D", "ERFM5D"):
            sse_rows.append({"model_id": mid, "space": space, "sse_per_customer": sse[mid][space]["sse_per_customer"]})
    sse_df = pd.DataFrame(sse_rows); stem = "fig06_common_space_sse_per_customer"; sources[stem] = write_source(stem, sse_df)
    scene = Scene("Common-space SSE per customer", "Stage 22 recomputed SSE on locked assignments; lower is better within the same space")
    labels = [f"{r.model_id}\n{r.space}" for r in sse_df.itertuples()]; vals = sse_df.sse_per_customer.astype(float).tolist(); bar_chart(stem, scene.title, scene.subtitle, labels, vals, "SSE per customer", value_fmt="{:.3f}")
    manifests.append((stem, "Common-space SSE", ["Stage 22"]))

    balance_rows = []
    for mid in ("B0", "M1", "M2"):
        balance_rows.append({"model_id": mid, "normalized_entropy": balance[mid]["normalized_entropy"], "cluster_size_cv": balance[mid]["cluster_size_cv"], "share_range_pp": balance[mid]["max_ratio_pct"] - balance[mid]["min_ratio_pct"]})
    bal = pd.DataFrame(balance_rows)
    for suffix, col, title, ylabel in (("a", "normalized_entropy", "Cluster entropy", "Normalized entropy"), ("b", "cluster_size_cv", "Cluster coefficient of variation", "CV"), ("c", "share_range_pp", "Cluster share range", "Percentage points")):
        substem = f"fig07{suffix}_cluster_{'entropy' if suffix=='a' else 'cv' if suffix=='b' else 'share_range'}"; frame = bal[["model_id", col]].rename(columns={col: "value"}); sources[substem] = write_source(substem, frame); bar_chart(substem, title, "Stage 22 locked cluster-balance metrics", frame.model_id.tolist(), frame.value.astype(float).tolist(), ylabel, value_fmt="{:.3f}"); manifests.append((substem, title, ["Stage 22"]))

    # 08-10: M2 segment layer.
    seg = q6.copy(); seg["prediction"] = seg["prediction"].astype(int); finite_frame(seg.select_dtypes(exclude=["object"]), "fig08")
    stem = "fig08_m2_segment_distribution"; sources[stem] = write_source(stem, seg[["prediction", "segment_name_en", "customer_count", "customer_ratio"]]); bar_chart(stem, "M2 segment distribution", "Stage 27 SQL readback of the locked Stage 16 membership authority", seg.segment_name_en.tolist(), seg.customer_count.astype(float).tolist(), "Customers", value_fmt="{:,.0f}"); manifests.append((stem, "M2 segment distribution", ["Stage 16", "Stage 27"]))
    median_cols = ["prediction", "median_recency", "median_frequency", "median_monetary", "median_avg_basket_size", "median_product_variety"]
    raw_medians = seg[median_cols].copy(); raw_medians.columns = ["segment", "Recency", "Frequency", "Monetary", "AvgBasketSize", "ProductVariety"]
    raw_medians.to_csv(DATA_DIR / "m2_segment_raw_medians.csv", index=False, encoding="utf-8-sig")
    normalized = raw_medians.copy()
    normalization_warnings = []
    for column in raw_medians.columns[1:]:
        lo, hi = float(raw_medians[column].min()), float(raw_medians[column].max())
        if hi == lo:
            normalized[column] = 0.0
            normalization_warnings.append(f"{column}: max equals min; normalized to 0.0")
        else:
            normalized[column] = (raw_medians[column] - lo) / (hi - lo)
    normalized.to_csv(DATA_DIR / "m2_segment_visual_normalized_medians.csv", index=False, encoding="utf-8-sig")
    prof = normalized.copy(); stem = "fig09_m2_segment_profile_heatmap"; sources[stem] = "data/m2_segment_visual_normalized_medians.csv"
    scene = Scene("M2 segment profile heatmap", "Locked segment medians with visualization-only min-max normalization")
    vals = prof.iloc[:, 1:].to_numpy(float); x0, y0 = 300, 180; cellw, cellh = 190, 80
    for i, name in enumerate(prof.columns[1:]): scene.text(x0 + i * cellw + cellw / 2, y0 - 42, name, size=16, bold=True, anchor="middle")
    for r, row in enumerate(vals):
        scene.text(x0 - 20, y0 + r * cellh + 28, f"S{int(prof.iloc[r,0])}", size=18, bold=True, anchor="end")
        for c, value in enumerate(row):
            color = "#D7E8F9" if float(value) < 0.5 else "#FAD8C5"; scene.rect(x0 + c * cellw, y0 + r * cellh, cellw - 4, cellh - 4, color); scene.text(x0 + c * cellw + cellw / 2, y0 + r * cellh + 25, f"{value:.2f}", size=18, anchor="middle")
    scene.save(stem); manifests.append((stem, "M2 profile heatmap", ["Stage 16", "Stage 27"]))
    radar_cols = list(prof.columns[1:]); radar_frame = prof.copy(); fig10_source_equal = radar_frame.equals(prof); stem = "fig10_m2_segment_radar"; sources[stem] = "data/m2_segment_visual_normalized_medians.csv"
    scene = Scene("M2 segment radar profiles", "The same normalized locked medians used by Figure 09")
    cx, cy, radius = 700, 425, 250; n = len(radar_cols); angles = [2 * math.pi * i / n - math.pi / 2 for i in range(n)]; scene.circle(cx, cy, radius, SURFACE, GRID, 2)
    for ring in (0.25, 0.5, 0.75, 1.0): scene.circle(cx, cy, radius * ring, SURFACE, GRID, 1)
    for angle, label in zip(angles, ["Recency", "Frequency", "Monetary", "Basket", "Variety"]):
        x, y = cx + radius * math.cos(angle), cy + radius * math.sin(angle); scene.line(cx, cy, x, y, GRID, 1); scene.text(cx + (radius + 35) * math.cos(angle), cy + (radius + 35) * math.sin(angle), label, size=16, anchor="middle")
    for r, row in radar_frame[radar_cols].iterrows():
        pts = [(cx + radius * float(v) * math.cos(a), cy + radius * float(v) * math.sin(a)) for v, a in zip(row, angles)]; scene.polyline(pts, COLORS[r % len(COLORS)], 3, closed=True); scene.text(1050, 230 + r * 35, f"S{int(radar_frame.iloc[r,0])}", size=16, fill=COLORS[r % len(COLORS)])
    scene.save(stem); manifests.append((stem, "M2 radar profiles", ["Stage 16", "Stage 27"]))

    # 11-16: agreement and system benchmarks.
    ag = pd.DataFrame([{"metric": "ARI", "value": agreement["ARI"]}, {"metric": "NMI", "value": agreement["NMI"]}, {"metric": "Mismatch ratio", "value": agreement["mismatch_ratio"]}]); stem = "fig11_b0_m1_agreement"; sources[stem] = write_source(stem, ag); bar_chart(stem, "B0 vs M1 agreement", "Comparison-only label alignment; locked B0 IDs remain unchanged", ag.metric.tolist(), ag.value.astype(float).tolist(), "Agreement / ratio", value_fmt="{:.4f}"); manifests.append((stem, "B0/M1 agreement", ["Stage 22"]))
    rt = stage23_b[["ModelID", "MedianFitOnlySec", "MedianAssignAfterFitSec", "MedianTotalFitPlusAssignSec"]].copy(); stem = "fig12_model_runtime_benchmark"; sources[stem] = write_source(stem, rt)
    scene = Scene("B0/M1/M2 runtime benchmark", "Stage 23 controlled 1M cohort; medians are machine- and cohort-specific")
    labels = rt["ModelID"].tolist(); x0, y0, x1, y1 = 150, 150, 1330, 680; ymax = float(rt[["MedianFitOnlySec", "MedianAssignAfterFitSec", "MedianTotalFitPlusAssignSec"]].max().max()) * 1.2
    sx, sy = axis(scene, x0, y0, x1, y1, 0, len(labels), 0, ymax, [(i + .5, l) for i, l in enumerate(labels)], [(float(v), f"{v:.0f}") for v in np.linspace(0, ymax, 5)], ylabel="Median seconds")
    bw = (x1 - x0) / len(labels) * .20
    for i, row in rt.iterrows():
        for j, col in enumerate(("MedianFitOnlySec", "MedianAssignAfterFitSec", "MedianTotalFitPlusAssignSec")):
            xx = sx(i + .5) + (j - 1) * bw - bw / 2; yy = sy(row[col]); scene.rect(xx, yy, bw, y1 - yy, COLORS[j]); scene.text(xx + bw / 2, yy - 24, f"{row[col]:.2f}", size=13, anchor="middle")
    for j, label in enumerate(("Fit", "Assignment", "Total")): scene.text(x1 - 10, y0 + 18 + j * 25, label, size=16, fill=COLORS[j], anchor="end")
    scene.save(stem); manifests.append((stem, "Model runtime benchmark", ["Stage 23"]))
    storage = pd.DataFrame([{"format": "CSV", "bytes": s24["storage"]["csv_bytes"]}, {"format": "Parquet", "bytes": s24["storage"]["parquet_bytes"]}]); stem = "fig13_csv_parquet_storage"; sources[stem] = write_source(stem, storage); bar_chart(stem, "CSV vs Parquet storage", "Stage 24 recursive data-file size comparison", storage.format.tolist(), storage.bytes.astype(float).tolist(), "Bytes", value_fmt="{:,.0f}"); manifests.append((stem, "Storage comparison", ["Stage 24"]))
    r24 = stage24_runtime[["QueryName", "CSVMedianSec", "ParquetMedianSec"]].copy(); stem = "fig14_csv_parquet_runtime"; sources[stem] = write_source(stem, r24); scene = Scene("CSV vs Parquet query runtime", "Stage 24 controlled repeated-read medians"); labels = r24.QueryName.tolist(); x0, y0, x1, y1 = 150, 150, 1330, 680; ymax = float(r24[["CSVMedianSec", "ParquetMedianSec"]].max().max()) * 1.2; sx, sy = axis(scene, x0, y0, x1, y1, 0, len(labels), 0, ymax, [(i+.5,l) for i,l in enumerate(labels)], [(float(v),f"{v:.0f}") for v in np.linspace(0,ymax,5)], ylabel="Median seconds"); bw=(x1-x0)/len(labels)*.28
    for i,row in r24.iterrows():
        for j,col in enumerate(("CSVMedianSec","ParquetMedianSec")): xx=sx(i+.5)+(j-.5)*bw-bw/2; yy=sy(row[col]); scene.rect(xx,yy,bw,y1-yy,COLORS[j]); scene.text(xx+bw/2,yy-25,f"{row[col]:.2f}",size=14,anchor="middle")
    scene.text(x1-10,y0+20,"CSV",size=16,fill=COLORS[0],anchor="end"); scene.text(x1-10,y0+48,"Parquet",size=16,fill=COLORS[1],anchor="end"); scene.save(stem); manifests.append((stem,"CSV/Parquet runtime",["Stage 24"]))
    r25 = stage25_runtime[["QueryID", "NoCacheMedianSec", "CacheMedianSec"]].copy(); stem = "fig15_cache_vs_no_cache"; sources[stem] = write_source(stem, r25); scene = Scene("Cache vs no-cache runtime", "Stage 25 controlled repeated-read medians; cache materialization is not included in the bars"); labels=r25.QueryID.tolist(); x0,y0,x1,y1=150,150,1330,680; ymax=float(r25[["NoCacheMedianSec","CacheMedianSec"]].max().max())*1.2; sx,sy=axis(scene,x0,y0,x1,y1,0,len(labels),0,ymax,[(i+.5,l) for i,l in enumerate(labels)],[(float(v),f"{v:.1f}") for v in np.linspace(0,ymax,5)],ylabel="Median seconds"); bw=(x1-x0)/len(labels)*.28
    for i,row in r25.iterrows():
        for j,col in enumerate(("NoCacheMedianSec","CacheMedianSec")): xx=sx(i+.5)+(j-.5)*bw-bw/2; yy=sy(row[col]); scene.rect(xx,yy,bw,y1-yy,COLORS[j]); scene.text(xx+bw/2,yy-25,f"{row[col]:.2f}",size=14,anchor="middle")
    scene.text(x1-10,y0+20,"No cache",size=16,fill=COLORS[0],anchor="end"); scene.text(x1-10,y0+48,"Cache",size=16,fill=COLORS[1],anchor="end"); scene.save(stem); manifests.append((stem,"Cache/no-cache runtime",["Stage 25"]))
    sh = pd.DataFrame([{ "query": q, "shuffle_read_bytes": shuffle[q]["shuffle_read_bytes"], "shuffle_write_bytes": shuffle[q]["shuffle_write_bytes"]} for q in ("Q2_FILTER_AGGREGATION", "Q3_CUSTOMER_AGGREGATION")]); stem="fig16_spark_shuffle_q2_vs_q3"; sources[stem]=write_source(stem,sh); bar_chart(stem,"Spark shuffle: Q2 vs Q3","Stage 26 execution evidence; Q3 is the customer aggregation path",sh["query"].tolist(),sh.shuffle_read_bytes.astype(float).tolist(),"Shuffle read bytes",value_fmt="{:,.0f}"); manifests.append((stem,"Spark shuffle evidence",["Stage 26"]))

    # 17-19: Stage 27 SQL analytics.
    q2["month_label"] = q2["month"].str.slice(0, 7); stem="fig17_monthly_monetary_trend"; source=q2[["month_label","monetary_sum"]].copy(); sources[stem]=write_source(stem,source); line_chart(stem,"Monthly monetary trend","Stage 27 Spark SQL analytical layer",list(range(len(source))),[("Monetary sum",source.monetary_sum.astype(float).tolist())],"Month","Monetary sum",source.month_label.tolist()); manifests.append((stem,"Monthly monetary trend",["Stage 27"]))
    source=q4.head(10)[["product_group","purchase_item_rows","monetary_sum"]].copy(); stem="fig18_top_product_groups"; sources[stem]=write_source(stem,source); bar_chart(stem,"Top product groups","Stage 27 SQL output; top ten by source order",source.product_group.tolist(),source.monetary_sum.astype(float).tolist(),"Monetary sum",value_fmt="{:.0f}"); manifests.append((stem,"Top product groups",["Stage 27"]))
    source=q5[["age_band","monetary_sum","average_monetary_per_customer"]].copy(); stem="fig19_age_band_monetary"; sources[stem]=write_source(stem,source); bar_chart(stem,"Monetary by age band","Stage 27 SQL output; total monetary sum",source.age_band.tolist(),source.monetary_sum.astype(float).tolist(),"Monetary sum",value_fmt="{:.0f}"); manifests.append((stem,"Age-band monetary",["Stage 27"]))

    # 20-23: FP-Growth, including preserved V1 negative point.
    v1 = pd.DataFrame([{ "version": "V1", "min_support": s28_v1_config["minSupport"], "frequent_itemset_count": s28_v1_items["frequent_itemset_count"], "pair_or_larger_itemset_count": 0, "association_rule_count": 0, "selected": False }])
    v2 = threshold.rename(columns={"MinSupport":"min_support", "FrequentItemsetCount":"frequent_itemset_count", "PairOrLargerItemsetCount":"pair_or_larger_itemset_count", "AssociationRuleCount":"association_rule_count", "Selected":"selected"})[["min_support","frequent_itemset_count","pair_or_larger_itemset_count","association_rule_count","selected"]].copy(); v2["version"]="V2"; v2=v2[["version","min_support","frequent_itemset_count","pair_or_larger_itemset_count","association_rule_count","selected"]]; sens=pd.concat([v1,v2],ignore_index=True); stem="fig20_fp_growth_threshold_sensitivity"; sources[stem]=write_source(stem,sens); scene=Scene("FP-Growth threshold sensitivity","Stage 28 V1 negative point plus Stage 28 V2 protocol ladder; selected row is derived from the current locked run"); x0,y0,x1,y1=160,150,1320,680; labels=[f"{v}:{s:.5f}" for v,s in zip(sens.version,sens.min_support)]; vals=sens.association_rule_count.astype(float).tolist(); bar_chart(stem,scene.title,scene.subtitle,labels,vals,"Association rules",value_fmt="{:,.0f}"); manifests.append((stem,"FP-Growth threshold sensitivity",["Stage 28 V1","Stage 28 V2"]))
    enriched_name_lookup = {}
    for rule in enriched_rules:
        a_key = tuple(rule.get("antecedent", [])); c_key = tuple(rule.get("consequent", []))
        a_names = " + ".join(p.get("prod_name", p.get("article_id", "")) for p in rule.get("antecedent_products", []))
        c_names = " + ".join(p.get("prod_name", p.get("article_id", "")) for p in rule.get("consequent_products", []))
        enriched_name_lookup[(a_key, c_key)] = f"{a_names} → {c_names}"
    def rule_ids(value):
        if isinstance(value, (list, tuple)): return tuple(str(v) for v in value)
        return tuple(str(value).split(","))
    def rules_chart(stem, title, frame, col, readable=False):
        use=frame.head(10).copy()
        use["rule_label"] = [enriched_name_lookup.get((rule_ids(a), rule_ids(b)), f"{a} → {b}") if readable else f"{a} → {b}" for a,b in zip(use.antecedent,use.consequent)]
        out=use[["antecedent","consequent","rule_label","support","confidence","lift"]]; sources[stem]=write_source(stem,out); bar_chart(stem,title,"Stage 28 V2 locked rules; article IDs remain in the source CSV",out["rule_label"].tolist(),out[col].astype(float).tolist(),col.title(),value_fmt="{:.3f}"); manifests.append((stem,title,["Stage 28 V2"]))
    rules_chart("fig21_top_fp_growth_rules_by_lift","Top FP-Growth rules by lift",rules_lift,"lift",readable=True)
    rules_chart("fig22_top_fp_growth_rules_by_confidence","Top FP-Growth rules by confidence",rules_conf,"confidence")
    support_conf=rules_all[["antecedent_str","consequent_str","support","confidence","lift"]].rename(columns={"antecedent_str":"antecedent","consequent_str":"consequent"}).copy(); stem="fig23_fp_growth_support_confidence"; sources[stem]=write_source(stem,support_conf); scene=Scene("FP-Growth support vs confidence","All 104 Stage 28 V2 selected rules; marker size is proportional to lift"); x0,y0,x1,y1=160,150,1320,680; xmin,xmax=float(support_conf["support"].min()),float(support_conf["support"].max()); xpad=(xmax-xmin)*0.12 or 0.00001; sx,sy=axis(scene,x0,y0,x1,y1,xmin-xpad,xmax+xpad,0,1,[(xmin,f"{xmin:.5f}"),(xmax,f"{xmax:.5f}")],[(0,"0"),(.5,"0.5"),(1,"1")],xlabel="Support",ylabel="Confidence")
    for i,row in support_conf.iterrows(): scene.circle(sx(float(row.support)),sy(float(row.confidence)),max(5,min(26,5+math.log1p(float(row.lift))*2)),COLORS[i%len(COLORS)])
    scene.save(stem); manifests.append((stem,"FP-Growth support/confidence",["Stage 28 V2"]))

    # 24: a static analytics story map for reports/presentations.
    story = pd.DataFrame([
        {"stage":"RFM/eRFM","metric":"Customer profiles","value":int(s16["customer_count"]),"note":"Stage 5/6/12 locked feature evidence"},
        {"stage":"M2","metric":"Segments","value":int(s16["selected_k"]),"note":"Stage 15/16 locked enhanced model"},
        {"stage":"Spark SQL","metric":"Transaction rows","value":int(s27["transaction_row_count"]),"note":"Stage 27 analytical layer"},
        {"stage":"FP-Growth","metric":"Association rules","value":int(s28["association_rule_count"]),"note":"Stage 28 V2 selected configuration"},
    ]); stem="fig24_project_analytics_story"; sources[stem]=write_source(stem,story); scene=Scene("Project analytics story","Report-ready path from customer features to segments, SQL analytics, and association rules"); x_positions=[180,500,820,1140]; y=390
    for i,row in story.iterrows():
        x=x_positions[i]; scene.circle(x,y,75,COLORS[i]); scene.text(x,y-13,row.stage,size=19,fill="#FFFFFF",bold=True,anchor="middle"); scene.text(x,y+105,row.metric,size=17,bold=True,anchor="middle"); scene.text(x,y+135,f"{row.value:,}",size=20,fill=COLORS[i],bold=True,anchor="middle");
        if i<len(story)-1: scene.line(x+80,y,x_positions[i+1]-80,y,COLORS[i],5)
    scene.save(stem); manifests.append((stem,"Project analytics story",["Stage 5/6/12/16/27/28"]))

    # Every physical figure has a manifest and a source CSV.
    required_stems = [
        "fig01_rfm_erfm_distributions","fig02_skewness_transformation","fig03a_main_silhouette_by_k","fig03b_main_wcss_by_k","fig04a_enhanced_silhouette_by_k","fig04b_enhanced_wcss_by_k","fig05_primary_model_quality_comparison","fig06_common_space_sse_per_customer","fig07a_cluster_entropy","fig07b_cluster_cv","fig07c_cluster_share_range","fig08_m2_segment_distribution","fig09_m2_segment_profile_heatmap","fig10_m2_segment_radar","fig11_b0_m1_agreement","fig12_model_runtime_benchmark","fig13_csv_parquet_storage","fig14_csv_parquet_runtime","fig15_cache_vs_no_cache","fig16_spark_shuffle_q2_vs_q3","fig17_monthly_monetary_trend","fig18_top_product_groups","fig19_age_band_monetary","fig20_fp_growth_threshold_sensitivity","fig21_top_fp_growth_rules_by_lift","fig22_top_fp_growth_rules_by_confidence","fig23_fp_growth_support_confidence","fig24_project_analytics_story"
    ]
    if sorted(sources) != sorted(required_stems):
        raise AssertionError(f"Figure/source inventory mismatch: {sorted(set(required_stems)-set(sources))} / {sorted(set(sources)-set(required_stems))}")
    def source_path_for(stem):
        return OUT / Path(sources[stem])
    for stem in required_stems:
        png, svg, csv_path = FIG_DIR / f"{stem}.png", FIG_DIR / f"{stem}.svg", source_path_for(stem)
        if not png.exists() or not svg.exists() or not csv_path.exists(): raise FileNotFoundError(f"Missing output for {stem}")

    # Value consistency gates use exact locked values, not visual approximations.
    selected_rows = threshold[threshold["Selected"] == True]
    consistency = {
        "authority": AUTHORITY_STAGE29,
        "stage28_selected_support_matches_source": len(selected_rows) == 1 and float(selected_rows.iloc[0]["MinSupport"]) == float(s28["selected_minSupport"]),
        "stage28_rule_count_matches_source": len(selected_rows) == 1 and int(selected_rows.iloc[0]["AssociationRuleCount"]) == int(s28["association_rule_count"]),
        "m2_segment_counts_sum_to_customer_count": int(q6["customer_count"].sum()) == int(s16["customer_count"]),
        "stage22_m2_silhouette_matches_common_space": abs(float(stage22_matrix.loc[stage22_matrix.ModelID == "M2", "Silhouette_ERFM5D"].iloc[0]) - float(silhouettes["M2"]["ERFM5D"]["silhouette"])) < 1e-12,
        "stage24_storage_values_match_summary": int(storage.loc[storage.format == "CSV", "bytes"].iloc[0]) == int(s24["storage"]["csv_bytes"]) and int(storage.loc[storage.format == "Parquet", "bytes"].iloc[0]) == int(s24["storage"]["parquet_bytes"]),
        "stage26_shuffle_values_match_summary": int(sh.loc[sh["query"] == "Q3_CUSTOMER_AGGREGATION", "shuffle_read_bytes"].iloc[0]) == int(shuffle["Q3_CUSTOMER_AGGREGATION"]["shuffle_read_bytes"]),
        "all_source_values_finite": True,
    }
    for stem in required_stems:
        frame = pd.read_csv(source_path_for(stem))
        finite_frame(frame, stem)
    consistency["fig09_uses_locked_segment_medians"] = list(raw_medians.columns) == ["segment", "Recency", "Frequency", "Monetary", "AvgBasketSize", "ProductVariety"] and (OUT / "data" / "m2_segment_raw_medians.csv").exists()
    consistency["fig09_uses_minmax_visual_normalization"] = all(float(normalized[column].min()) >= -1e-12 and float(normalized[column].max()) <= 1 + 1e-12 for column in normalized.columns[1:])
    normalized_readback = pd.read_csv(OUT / "data" / "m2_segment_visual_normalized_medians.csv")
    consistency["fig09_fig10_normalized_source_equal"] = fig10_source_equal and list(normalized_readback.columns) == list(normalized.columns) and np.array_equal(normalized_readback["segment"].astype(int).to_numpy(), normalized["segment"].astype(int).to_numpy()) and np.allclose(normalized_readback.iloc[:, 1:].to_numpy(float), normalized.iloc[:, 1:].to_numpy(float), rtol=0.0, atol=1e-12)
    consistency["fig23_rule_count"] = int(len(rules_all))
    consistency["fig23_rule_count_matches_stage28"] = int(len(rules_all)) == int(s28["association_rule_count"])
    consistency["fig12_contains_fit_assignment_total"] = all(column in rt.columns for column in ("MedianFitOnlySec", "MedianAssignAfterFitSec", "MedianTotalFitPlusAssignSec"))

    key_metric_rows = [
        ("dataset_transaction_rows", q1_kpi["transaction_item_rows"], "Stage 27 q1_dataset_kpi.json"),
        ("customer_count", s16["customer_count"], "Stage 16 final profile summary"),
        ("distinct_articles", q1_kpi["distinct_articles"], "Stage 27 q1_dataset_kpi.json"),
        ("m1_selected_k", s9["selected_k"], "Stage 9 optimal_k_selection.json"),
        ("m2_selected_k", s16["selected_k"], "Stage 16 final profile summary"),
        ("m1_rfm3d_silhouette", silhouettes["M1"]["RFM3D"]["silhouette"], "Stage 22 common_space_silhouette.json"),
        ("m2_erfm5d_silhouette", silhouettes["M2"]["ERFM5D"]["silhouette"], "Stage 22 common_space_silhouette.json"),
        ("m2_min_cluster_share", balance["M2"]["min_ratio_pct"] / 100.0, "Stage 22 cluster_balance_metrics.json"),
        ("m2_max_cluster_share", balance["M2"]["max_ratio_pct"] / 100.0, "Stage 22 cluster_balance_metrics.json"),
        ("parquet_storage_reduction_pct", s24["storage"]["storage_reduction_pct"], "Stage 24 storage_size_comparison.json"),
        ("csv_parquet_q1_runtime_ratio", s24["queries"]["Q1"]["csv_over_parquet_ratio"], "Stage 24 runtime_summary.json"),
        ("q3_cache_break_even_executions", float(stage25_runtime.loc[stage25_runtime["QueryID"] == "Q3_CUSTOMER_AGGREGATION", "BreakEvenExecutions"].iloc[0]), "Stage 25 cache_vs_no_cache_summary.csv"),
        ("stage26_q3_shuffle_bytes", shuffle["Q3_CUSTOMER_AGGREGATION"]["shuffle_read_bytes"], "Stage 26 shuffle_metrics.json"),
        ("fp_growth_selected_minSupport", s28["selected_minSupport"], "Stage 28 V2 final summary"),
        ("fp_growth_frequent_itemset_count", s28["frequent_itemset_count"], "Stage 28 V2 final summary"),
        ("fp_growth_association_rule_count", s28["association_rule_count"], "Stage 28 V2 final summary"),
    ]
    key_metrics = pd.DataFrame([{"metric": metric, "value": value, "source": source} for metric, value, source in key_metric_rows]); key_metrics.to_csv(OUT / "final_key_metrics.csv", index=False, encoding="utf-8-sig")
    required_key_metrics = [name for name, _, _ in key_metric_rows]
    consistency["final_key_metrics_complete"] = set(required_key_metrics) == set(key_metrics["metric"].tolist())
    consistency["visual_value_consistency_pass"] = all(bool(v) for v in consistency.values() if isinstance(v, (bool, np.bool_)))
    write_json(OUT / "visual_value_consistency.json", consistency)
    presentation = {"authority": AUTHORITY_STAGE29, "recommended_for_report": ["fig05_primary_model_quality_comparison","fig08_m2_segment_distribution","fig09_m2_segment_profile_heatmap","fig17_monthly_monetary_trend","fig20_fp_growth_threshold_sensitivity","fig24_project_analytics_story"], "recommended_for_presentation": ["fig24_project_analytics_story","fig05_primary_model_quality_comparison","fig08_m2_segment_distribution","fig18_top_product_groups","fig20_fp_growth_threshold_sensitivity","fig21_top_fp_growth_rules_by_lift"], "selection_note":"Recommendations are presentation-layer choices only; no source metrics are changed."}
    write_json(OUT / "presentation_figure_selection.json", presentation)

    def manifest_meta(stem):
        if stem.startswith("fig01"):
            return "Stages 5/6/12", ["IMPROVED_ERFM_FEATURE_SPACE"], ["output/analysis/rfm_distribution/summary.json", "output/analysis/erfm_distribution/summary.json", "output/analysis/stage12/transformation_summary.json"], "quantile profile", "Displays locked RFM/eRFM quantile ranges and medians.", []
        if stem.startswith("fig02"):
            return "Stage 12", ["IMPROVED_ERFM_FEATURE_SPACE"], ["output/analysis/stage12/skewness_comparison.json", "output/analysis/stage12/transformation_summary.json"], "grouped bar", "Shows raw versus log1p skewness evidence.", []
        if stem.startswith("fig03"):
            return "Stage 8", [], ["output/analysis/stage8/kmeans_k_sweep.csv"], "line chart", "Shows the locked main-model K sweep.", []
        if stem.startswith("fig04"):
            return "Stage 13", [s13["authority"]], ["output/analysis/stage13/enhanced_kmeans_k_sweep.csv"], "line chart", "Shows the locked enhanced-model K sweep.", []
        if stem.startswith("fig05") or stem.startswith("fig06") or stem.startswith("fig07") or stem.startswith("fig11"):
            return "Stage 22", [s22["authority"]], ["output/comparison/stage22/primary_three_model_comparison_matrix.csv", "output/comparison/stage22/common_space_silhouette.json", "output/comparison/stage22/common_space_sse.json", "output/comparison/stage22/cluster_balance_metrics.json", "output/comparison/stage22/b0_m1_label_alignment.json"], "comparison chart", "Compares locked model quality, common-space SSE, balance, or agreement.", []
        if stem.startswith("fig08") or stem.startswith("fig09") or stem.startswith("fig10"):
            return "Stages 16/27", [s16["authority"], s27["authority"]], ["output/analysis/stage16/cluster_profiles.csv", "output/sql/stage27/q6_m2_segment_analytics.csv"], "segment profile chart", "Profiles the locked M2 segments without changing membership semantics.", normalization_warnings if stem.startswith("fig09") else []
        if stem.startswith("fig12"):
            return "Stage 23", [s23["authority"]], ["output/benchmark/stage23/track_b_build_assign_summary.csv"], "grouped bar chart", "Shows fit, assignment, and total runtime medians for the controlled cohort.", ["Runtime is cohort- and machine-specific."]
        if stem.startswith("fig13") or stem.startswith("fig14"):
            return "Stage 24", [s24["authority"]], ["output/benchmark/stage24/storage_size_comparison.json", "output/benchmark/stage24/csv_vs_parquet_benchmark_summary.csv"], "bar chart", "Shows locked CSV/Parquet storage or runtime evidence.", ["This is a controlled repeated-read benchmark."]
        if stem.startswith("fig15"):
            return "Stage 25", [s25["authority"]], ["output/benchmark/stage25/cache_vs_no_cache_summary.csv"], "grouped bar chart", "Shows locked cache versus no-cache runtime medians.", ["Cache behavior is workload-specific."]
        if stem.startswith("fig16"):
            return "Stage 26", [s26["authority"]], ["output/analysis/stage26/shuffle_metrics.json"], "bar chart", "Shows the locked Q2/Q3 shuffle-read contrast.", []
        if stem.startswith("fig17") or stem.startswith("fig18") or stem.startswith("fig19"):
            return "Stage 27", [s27["authority"]], ["output/sql/stage27/q2_monthly_purchase_trend.csv", "output/sql/stage27/q4_top20_product_groups.csv", "output/sql/stage27/q5_age_band_analysis.csv"], "SQL analytical chart", "Visualizes locked Spark SQL analytical outputs.", []
        if stem.startswith("fig20"):
            return "Stages 28 V1/V2", ["FP_GROWTH_ASSOCIATION_RULES_V1", AUTHORITY_STAGE28], ["output/fpgrowth/stage28/frequent_itemset_summary.json", "output/fpgrowth/stage28_v2/threshold_sensitivity.csv"], "threshold sensitivity bar chart", "Shows the preserved V1 negative point and V2 threshold protocol.", []
        if stem.startswith("fig21") or stem.startswith("fig22") or stem.startswith("fig23"):
            return "Stage 28 V2", [AUTHORITY_STAGE28], ["output/fpgrowth/stage28_v2/selected_association_rules.parquet", "output/fpgrowth/stage28_v2/enriched_top_rules.json"], "association-rule chart", "Visualizes locked FP-Growth rule support, confidence, or lift.", []
        return "Stages 5/6/12/16/27/28", ["IMPROVED_ERFM_FEATURE_SPACE", s16["authority"], s27["authority"], AUTHORITY_STAGE28], [], "story diagram", "Connects the locked project analytical layers.", []
    manifest = []
    manifest_required = ["figure_id", "title", "png_path", "svg_path", "source_stage", "source_authority", "source_artifacts", "source_data_csv", "chart_type", "scientific_interpretation", "warnings", "readback_required", "report_ready"]
    for stem, title, _ in manifests:
        source_stage, source_authority, source_artifacts, chart_type, interpretation, fig_warnings = manifest_meta(stem)
        manifest.append({"figure_id": stem, "title": title, "png_path": f"figures/{stem}.png", "svg_path": f"figures/{stem}.svg", "source_stage": source_stage, "source_authority": source_authority, "source_artifacts": source_artifacts, "source_data_csv": sources[stem], "chart_type": chart_type, "scientific_interpretation": interpretation, "warnings": fig_warnings, "readback_required": True, "report_ready": True})
    figure_manifest_schema_complete = len(manifest) == len(required_stems) and all(all(key in entry for key in manifest_required) for entry in manifest)
    write_json(MANIFEST_DIR / "figure_manifest.json", manifest)
    consistency["figure_manifest_schema_complete"] = bool(figure_manifest_schema_complete)
    consistency["visual_value_consistency_pass"] = all(bool(v) for v in consistency.values() if isinstance(v, (bool, np.bool_)))
    write_json(OUT / "visual_value_consistency.json", consistency)

    warnings = list(s23.get("warnings", []))[:2] + list(s24.get("warnings", []))[:2] + ["Stage 29 creates no new Spark application; visuals are deterministic renderings of locked artifacts."]
    recommendations = [
        "Use fig05 and fig06 together when explaining why M2 is evaluated in its enhanced eRFM5D space rather than by raw cross-space SSE.",
        "Use fig08-10 for the business segmentation narrative: size, profile contrasts, and behavioral dimensions.",
        "Use fig17-19 for the Spark SQL analytical layer and fig20-23 for the association-rule story.",
        "Use SVG files in reports/presentation software when available; use 300-DPI PNGs for raster export."
    ]
    report_lines = [
        "STAGE 29 — FINAL VISUALIZATION LAYER", "", f"authority = {AUTHORITY_STAGE29}", f"generated_utc = {datetime.now(timezone.utc).isoformat()}", "scope = visualization-only; no Stage 1-28 source artifact modified", "",
        "A. AUTHORITY CHAIN", f"Stage 12 = {s12['authority']}", f"Stage 16 = {s16['authority']}", f"Stage 22 = {s22['authority']}", f"Stage 23 = {s23['authority']}", f"Stage 24 = {s24['authority']}", f"Stage 25 = {s25['authority']}", f"Stage 26 = {s26['authority']}", f"Stage 27 = {s27['authority']}", f"Stage 28 V2 = {s28['authority']}", "",
        "B. LOCKED KEY METRICS", f"M2 selected K = {s16['selected_k']}", f"customers = {s16['customer_count']}", f"Stage 28 selected minSupport = {s28['selected_minSupport']}", f"frequent itemsets = {s28['frequent_itemset_count']}", f"pair-or-larger itemsets = {s28['pair_or_larger_itemset_count']}", f"association rules = {s28['association_rule_count']}", "",
        "C. FIGURE INVENTORY", f"physical figure count = {len(required_stems)}", "PNG and SVG generated for every required figure; source CSV and manifest generated for every figure.", "",
        "D. BUSINESS STORY", "The visual layer connects skew-aware customer features, enhanced segmentation, SQL analytics, system behavior, and FP-Growth rules without changing model semantics.", "",
        "E. RFM/eRFM", "fig01 shows locked quantile profiles; fig02 shows the Stage 12 log1p skewness evidence.", "",
        "F. MODEL QUALITY", "fig03-07 cover K sweeps, primary comparison, common-space SSE, and cluster balance/agreement.", "",
        "G. SEGMENTS", "fig08-10 show M2 distribution and behavioral profiles from the Stage 16/27 locked membership outputs.", "",
        "H. SYSTEM BENCHMARKS", "fig11-16 show agreement, runtime, storage, cache, and shuffle evidence from Stages 22-26.", "",
        "I. SPARK SQL", "fig17-19 show monthly monetary trend, product groups, and age-band monetary analysis.", "",
        "J. FP-GROWTH", "fig20 preserves the V1 negative point and V2 threshold sensitivity; fig21-23 show locked rule quality.", "",
        "K. RECOMMENDATIONS", *[f"- {r}" for r in recommendations], "", "warnings =", *[f"- {w[:300]}" for w in warnings], "blockers = []"
    ]
    REPORT.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    # First-pass readback excludes the log which is created explicitly below.
    artifact_paths = [OUT / "visual_value_consistency.json", OUT / "final_key_metrics.csv", OUT / "presentation_figure_selection.json", MANIFEST_DIR / "figure_manifest.json", REPORT]
    artifact_paths += [FIG_DIR / f"{stem}.{ext}" for stem in required_stems for ext in ("png", "svg")]
    artifact_paths += sorted({source_path_for(stem) for stem in required_stems})
    artifact_paths += [DATA_DIR / "m2_segment_raw_medians.csv", DATA_DIR / "m2_segment_visual_normalized_medians.csv"]
    artifact_readback = {str(p.relative_to(BASE)): p.exists() and p.stat().st_size > 0 for p in artifact_paths}
    artifact_readback_pass = all(artifact_readback.values()) and consistency["visual_value_consistency_pass"]
    summary = {
        "stage": 29, "authority": AUTHORITY_STAGE29, "stage28_authority": AUTHORITY_STAGE28, "stage28_v2_final_summary": "output/fpgrowth/stage28_v2/stage28_v2_final_summary.json",
        "figure_count": len(required_stems), "required_figures_complete": all((FIG_DIR / f"{stem}.png").exists() for stem in required_stems), "source_data_complete": all(source_path_for(stem).exists() for stem in required_stems), "figure_source_data_complete": all(source_path_for(stem).exists() for stem in required_stems),
        "visual_value_consistency_pass": bool(consistency["visual_value_consistency_pass"]), "m2_segment_reconciliation_pass": bool(consistency["m2_segment_counts_sum_to_customer_count"]), "m2_cluster_count_reconciliation": bool(consistency["m2_segment_counts_sum_to_customer_count"]), "fig09_fig10_profile_consistency_pass": bool(consistency["fig09_fig10_normalized_source_equal"]), "fig23_all_rules_visualized": bool(consistency["fig23_rule_count_matches_stage28"]),
        "models_modified": False, "stage22_modified": False, "stage23_modified": False, "stage24_modified": False, "stage25_modified": False, "stage26_modified": False, "stage27_modified": False, "stage28_modified": False, "stage28_v2_modified": False, "stage28_v1_modified": False, "current_run_clean_namespace": clean_namespace,
        "artifact_readback": artifact_readback, "artifact_readback_pass": artifact_readback_pass, "final_summary_self_readback_pass": False, "visualization_layer_complete": False, "complete": False, "ready_for_stage30": False, "warnings": warnings, "blockers": []
    }
    summary_path = OUT / "stage29_final_summary.json"; write_json(summary_path, summary)
    summary_check = json.loads(summary_path.read_text(encoding="utf-8")); summary["final_summary_self_readback_pass"] = summary_check.get("authority") == AUTHORITY_STAGE29 and summary_check.get("figure_count") == len(required_stems)
    summary["complete"] = bool(summary["final_summary_self_readback_pass"] and summary["artifact_readback_pass"] and summary["visual_value_consistency_pass"] and not summary["blockers"])
    summary["visualization_layer_complete"] = summary["complete"]
    summary["ready_for_stage30"] = summary["complete"]

    # Explicitly create/overwrite the current Stage 29 run log, then include it in readback.
    def write_run_log():
        lines = [
            "stage = 29", f"authority = {AUTHORITY_STAGE29}", f"generated_utc = {datetime.now(timezone.utc).isoformat()}", "spark_application_id = NO_NEW_SPARK_APPLICATION_VISUALIZATION_ONLY", f"figure_count = {len(required_stems)}", f"fig09_fig10_profile_consistency_pass = {summary['fig09_fig10_profile_consistency_pass']}", f"fig23_rule_count = {consistency['fig23_rule_count']}", f"fig23_all_rules_visualized = {summary['fig23_all_rules_visualized']}", f"required_figures_complete = {summary['required_figures_complete']}", f"source_data_complete = {summary['source_data_complete']}", f"visual_value_consistency_pass = {summary['visual_value_consistency_pass']}", f"artifact_readback_pass = {summary['artifact_readback_pass']}", f"final_summary_self_readback_pass = {summary['final_summary_self_readback_pass']}", f"visualization_layer_complete = {summary['visualization_layer_complete']}", "blockers = []", f"ready_for_stage30 = {summary['ready_for_stage30']}",
        ]
        RUN_LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(summary_path, summary); write_run_log()
    artifact_readback[str(RUN_LOG.relative_to(BASE))] = RUN_LOG.exists() and RUN_LOG.stat().st_size > 0
    summary["artifact_readback"] = artifact_readback; summary["artifact_readback_pass"] = all(artifact_readback.values()) and consistency["visual_value_consistency_pass"]
    summary["complete"] = bool(summary["final_summary_self_readback_pass"] and summary["artifact_readback_pass"] and summary["visual_value_consistency_pass"] and not summary["blockers"]); summary["visualization_layer_complete"] = summary["complete"]; summary["ready_for_stage30"] = summary["complete"]
    write_json(summary_path, summary); summary_check = json.loads(summary_path.read_text(encoding="utf-8")); summary["final_summary_self_readback_pass"] = summary_check.get("authority") == AUTHORITY_STAGE29 and summary_check.get("figure_count") == len(required_stems) and summary_check.get("artifact_readback_pass") is True; summary["complete"] = bool(summary["final_summary_self_readback_pass"] and summary["artifact_readback_pass"] and summary["visual_value_consistency_pass"] and not summary["blockers"]); summary["visualization_layer_complete"] = summary["complete"]; summary["ready_for_stage30"] = summary["complete"]
    write_json(summary_path, summary); write_run_log()
    if not summary["complete"]:
        raise RuntimeError("Stage 29 final summary failed closed validation")
    return summary


if __name__ == "__main__":
    try:
        result = main()
        print(json.dumps({"stage": 29, "figure_count": result["figure_count"], "artifact_readback_pass": result["artifact_readback_pass"], "ready_for_stage30": result["ready_for_stage30"]}, ensure_ascii=False))
    except Exception as exc:
        try:
            if not RUN_LOG.exists():
                RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
                RUN_LOG.write_text("stage = 29\nauthority = FINAL_VISUALIZATION_LAYER_V1\nartifact_readback_pass = False\nfinal_summary_self_readback_pass = False\nready_for_stage30 = False\nblockers = [" + repr(str(exc)) + "]\n", encoding="utf-8")
        except Exception:
            pass
        traceback.print_exc()
        raise
