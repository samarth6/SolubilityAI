import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

COLORS = ["#339af0", "#9775fa", "#f59f00", "#51cf66", "#ff6b6b"]

def render_comparison_tab(cfg, trained_models, scaler, metrics, X, y):
    st.subheader("Model Performance Comparison")
    st.caption("All models trained on the same 30-molecule dataset.")

    metric_df = pd.DataFrame(metrics).T.reset_index()
    metric_df.columns = ["Model","R2","RMSE","CV_R2","CV_Std","MAE"]
    model_names = metric_df["Model"].tolist()
    colors = COLORS[:len(model_names)]

    fig = make_subplots(rows=1, cols=3,
        subplot_titles=("R² Score  (higher = better)",
                        "RMSE  (lower = better)",
                        "MAE  (lower = better)"),
        horizontal_spacing=0.1)

    for key, col in [("R2",1),("RMSE",2),("MAE",3)]:
        fig.add_trace(go.Bar(
            x=model_names, y=metric_df[key].tolist(),
            marker_color=colors,
            text=[str(v) for v in metric_df[key].tolist()],
            textposition="outside",
            textfont=dict(color="#212529", size=10),
            showlegend=False,
        ), row=1, col=col)

    fig.update_layout(height=340, margin=dict(t=50,b=10),
        paper_bgcolor="white", plot_bgcolor="white", font=dict(color="#212529"))
    fig.update_xaxes(gridcolor="#e9ecef", tickangle=-20)
    fig.update_yaxes(gridcolor="#e9ecef")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Full Metrics Table**")
    st.dataframe(metric_df.set_index("Model"), use_container_width=True)

    st.markdown(f"**Parity Plot — {cfg['model_name']}**")
    st.caption("Dots close to the red line = accurate predictions")
    preds = trained_models[cfg["model_name"]].predict(scaler.transform(X))
    lo = min(y.min(), preds.min()) - 0.5
    hi = max(y.max(), preds.max()) + 0.5

    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(x=y, y=preds, mode="markers",
        marker=dict(color="#339af0", size=9, opacity=0.75),
        hovertemplate="Actual: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>",
        name="Predictions"))
    fig_p.add_trace(go.Scatter(x=[lo,hi], y=[lo,hi], mode="lines",
        line=dict(color="#c92a2a", dash="dash", width=1.5),
        name="Perfect prediction"))
    fig_p.update_layout(height=380, paper_bgcolor="white", plot_bgcolor="white",
        font=dict(color="#212529"),
        xaxis=dict(title="Actual logS", gridcolor="#e9ecef"),
        yaxis=dict(title="Predicted logS", gridcolor="#e9ecef"),
        margin=dict(t=10, b=10))
    st.plotly_chart(fig_p, use_container_width=True)
