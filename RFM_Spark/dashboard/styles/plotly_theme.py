from __future__ import annotations

MODEL_COLORS = {
    "B0": "#5B6770",
    "M1": "#000080",
    "M2": "#581845",
}

SPACE_COLORS = {
    "RFM3D": "#000080",
    "eRFM5D": "#E8C3BA",
}

PALETTE = ["#000080", "#581845", "#E8C3BA", "#5B6770", "#F0EAD6"]


def apply_theme(fig, height: int = 400):
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=58, r=28, t=96, b=58),
        font=dict(family="Segoe UI, Arial, sans-serif", color="#1C1C1E", size=12),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        hoverlabel=dict(bgcolor="#FFFFFF", font_color="#1C1C1E"),
        title=dict(
            x=0,
            xanchor="left",
            y=0.98,
            yanchor="top",
            font=dict(size=15, color="#1C1C1E"),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,
            xanchor="left",
            x=0,
            bgcolor="rgba(255,255,255,0.92)",
        ),
    )
    fig.update_xaxes(showgrid=False, linecolor="#E0E0E0", zeroline=False)
    fig.update_yaxes(gridcolor="#E0E0E0", gridwidth=0.6, zeroline=False)
    return fig
