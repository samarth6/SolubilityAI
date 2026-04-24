import numpy as np
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

@st.cache_data
def compute_pca(X):
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    return coords, pca.explained_variance_ratio_

@st.cache_data
def compute_tsne(X, perplexity=10):
    perp = min(perplexity, len(X) - 1)
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, n_iter=500)
    return tsne.fit_transform(X)

def project_query(X_train, query):
    pca = PCA(n_components=2, random_state=42)
    pca.fit(X_train)
    return pca.transform(query)
