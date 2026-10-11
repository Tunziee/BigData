"""Plotly chart helpers used by the seven dashboard pages."""

from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go


COLORS = ["#2563eb", "#0f766e", "#f59e0b", "#dc2626", "#7c3aed", "#475569"]


def polish(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        height=height,
        template="plotly_white",
        margin=dict(l=16, r=16, t=42, b=16),
        legend_title_text="",
        font=dict(family="Arial", size=12),
    )
    return fig


def grouped_bar(df, x, y, color, title, labels=None, barmode="group"):
    return polish(px.bar(df, x=x, y=y, color=color, barmode=barmode, title=title, labels=labels, color_discrete_sequence=COLORS))


def line(df, x, y, title, labels=None):
    return polish(px.line(df, x=x, y=y, markers=True, title=title, labels=labels, color_discrete_sequence=COLORS))


def donut(df, names, values, title):
    return polish(px.pie(df, names=names, values=values, hole=0.52, title=title, color_discrete_sequence=COLORS), 390)


def heatmap(df, x, y, z, title):
    wide = df.pivot(index=y, columns=x, values=z)
    return polish(px.imshow(wide, text_auto=".2f", aspect="auto", title=title, color_continuous_scale="Blues"), 390)


def radar(row, dimensions):
    values = [float(row[d]) for d in dimensions]
    fig = go.Figure(go.Scatterpolar(r=values + [values[0]], theta=dimensions + [dimensions[0]], fill="toself", line_color=COLORS[0]))
    fig.update_layout(title="Hồ sơ trung vị cụm đã chọn", polar=dict(radialaxis=dict(visible=True)), showlegend=False)
    return polish(fig, 390)
