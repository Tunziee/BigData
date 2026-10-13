from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go

from styles.plotly_theme import MODEL_COLORS, PALETTE, SPACE_COLORS, apply_theme


def model_bar(df, x: str, y: str, title: str, color: str | None = None):
    fig = px.bar(df, x=x, y=y, color=color, barmode="group", title=title, color_discrete_map=MODEL_COLORS, color_discrete_sequence=PALETTE)
    return apply_theme(fig)


def space_bar(df, x: str, y: str, title: str):
    fig = px.bar(df, x=x, y=y, color="Space", barmode="group", title=title, color_discrete_map=SPACE_COLORS)
    return apply_theme(fig)


def donut(df, names: str, values: str, title: str):
    fig = px.pie(df, names=names, values=values, hole=0.55, title=title, color_discrete_sequence=PALETTE)
    fig = apply_theme(fig, 380)
    fig.update_layout(
        margin=dict(l=24, r=210, t=62, b=30),
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.92)",
        ),
    )
    return fig


def heatmap(df, x: str, y: str, z: str, title: str):
    matrix = df.pivot(index=y, columns=x, values=z)
    fig = px.imshow(matrix, text_auto=".2f", aspect="auto", color_continuous_scale=[[0, "#F0EAD6"], [0.5, "#E8C3BA"], [1, "#000080"]], title=title)
    return apply_theme(fig, 390)


def radar(df, dimensions: list[str], color_col: str, title: str):
    fig = go.Figure()
    for _, row in df.iterrows():
        values = [float(row[d]) for d in dimensions]
        values.append(values[0])
        theta = dimensions + [dimensions[0]]
        fig.add_trace(go.Scatterpolar(r=values, theta=theta, fill="toself", name=str(row[color_col])))
    fig.update_layout(title=title, polar=dict(radialaxis=dict(visible=True, gridcolor="#E0E0E0")))
    return apply_theme(fig, 430)
