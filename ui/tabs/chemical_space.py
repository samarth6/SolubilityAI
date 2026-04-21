import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from core.data        import get_name_from_smiles
from core.features    import extract_features
from core.projections import compute_pca, compute_tsne, project_query

def render_chemical_space_tab(cfg, scaler, X, y, df):
    st.subheader("Chemical Space Explorer")
    st.caption(
        "Each dot = a training molecule coloured by logS. "
        "Green = more soluble, Red = less soluble. ⭐ = your selected molecule."
    )

    method = st.radio("Projection", ["PCA","t-SNE"], horizontal=True)
    pca_coords, var_ratio = compute_pca(X)
    tsne_coords = compute_tsne(X, perplexity=cfg["perplexity"])

    if method == "PCA":
        coords = pca_coords
        xlabel = f"PC1 ({var_ratio[0]:.1%} variance)"
        ylabel = f"PC2 ({var_ratio[1]:.1%} variance)"
        title  = f"PCA — {var_ratio.sum():.1%} variance explained"
    else:
        coords = tsne_coords
        xlabel, ylabel = "Dimension 1", "Dimension 2"
        title = f"t-SNE (perplexity={cfg['perplexity']})"

    query_X, _ = extract_features([cfg["smiles"]])
    q_proj   = project_query(X, query_X) if len(query_X) > 0 else None
    mol_name = get_name_from_smiles(cfg["smiles"])
    names    = df["name"].tolist()
    smiles_l = df["smiles"].tolist()
    hover    = [f"<b>{n}</b><br>{s}<br>logS: {v:.2f}"
                for n,s,v in zip(names,smiles_l,y)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=coords[:,0], y=coords[:,1], mode="markers",
        marker=dict(color=y, colorscale="RdYlGn", size=11, opacity=0.85,
                    colorbar=dict(title="logS"),
                    line=dict(width=0.8, color="white")),
        text=hover, hovertemplate="%{text}<extra></extra>",
        name="Training molecules"))
    if q_proj is not None:
        fig.add_trace(go.Scatter(
            x=q_proj[:,0], y=q_proj[:,1], mode="markers+text",
            marker=dict(symbol="star", size=20, color="#f59f00",
                        line=dict(width=1.5, color="black")),
            text=[mol_name], textposition="top center",
            textfont=dict(color="#333", size=11),
            name=mol_name,
            hovertemplate=f"<b>{mol_name}</b><extra></extra>"))
    fig.update_layout(title=title, height=480,
        paper_bgcolor="white", plot_bgcolor="white", font=dict(color="#212529"),
        xaxis=dict(title=xlabel, gridcolor="#e9ecef", zeroline=False),
        yaxis=dict(title=ylabel, gridcolor="#e9ecef", zeroline=False),
        margin=dict(t=50,b=15))
    st.plotly_chart(fig, use_container_width=True)

    if method == "PCA":
        var_df = pd.DataFrame({
            "Component":      ["PC 1","PC 2"],
            "Variance (%)":   [f"{v*100:.2f}%" for v in var_ratio],
            "Cumulative (%)": [f"{var_ratio[:i+1].sum()*100:.2f}%" for i in range(2)],
        })
        st.dataframe(var_df.set_index("Component"), use_container_width=True)
