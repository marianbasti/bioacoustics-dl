
import streamlit as st
import torch
import plotly.express as px
import pandas as pd
from vis import prepare_features, reduce_dimensions
import numpy as np

@st.cache_data
def load_features(data_dir, checkpoint_path, batch_size):
    """Cache the feature extraction to avoid recomputing"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    features = prepare_features(data_dir, checkpoint_path, batch_size, device)
    return features.numpy() if isinstance(features, torch.Tensor) else features

def create_plot(embedded, method, params_str):
    """Create an interactive scatter plot using plotly"""
    df = pd.DataFrame(
        embedded,
        columns=[f'{method}_1', f'{method}_2']
    )
    
    fig = px.scatter(
        df,
        x=f'{method}_1',
        y=f'{method}_2',
        title=f'{method.upper()} Visualization ({params_str})',
        color=np.arange(len(df)),
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        width=600,
        height=600,
        coloraxis_colorbar_title='Sample Index'
    )
    
    return fig

def main():
    st.set_page_config(layout="wide", page_title="BEATs Feature Visualization")
    st.title("Interactive Audio Feature Visualization")

    # Sidebar for input parameters
    with st.sidebar:
        st.header("Parameters")
        data_dir = st.text_input("Data Directory", "path/to/audio/files")
        checkpoint_path = st.text_input("Checkpoint Path", "path/to/checkpoint.pt")
        batch_size = st.number_input("Batch Size", min_value=1, value=32)
        
        st.divider()
        st.header("t-SNE Parameters")
        perplexity = st.slider("Perplexity", min_value=5, max_value=100, value=30)
        
        st.divider()
        st.header("UMAP Parameters")
        n_neighbors = st.slider("Number of Neighbors", min_value=2, max_value=100, value=15)
        min_dist = st.slider("Minimum Distance", min_value=0.0, max_value=1.0, value=0.1, step=0.05)

    # Load features (cached)
    try:
        features = load_features(data_dir, checkpoint_path, batch_size)
    except Exception as e:
        st.error(f"Error loading features: {str(e)}")
        return

    # Create two columns for visualizations
    col1, col2 = st.columns(2)

    with col1:
        st.header("t-SNE Visualization")
        try:
            tsne_embedded, tsne_params = reduce_dimensions(
                features, 
                method='tsne',
                perplexity=perplexity
            )
            fig_tsne = create_plot(tsne_embedded, 't-SNE', tsne_params)
            st.plotly_chart(fig_tsne)
        except Exception as e:
            st.error(f"Error in t-SNE: {str(e)}")

    with col2:
        st.header("UMAP Visualization")
        try:
            umap_embedded, umap_params = reduce_dimensions(
                features,
                method='umap',
                n_neighbors=n_neighbors,
                min_dist=min_dist
            )
            fig_umap = create_plot(umap_embedded, 'UMAP', umap_params)
            st.plotly_chart(fig_umap)
        except Exception as e:
            st.error(f"Error in UMAP: {str(e)}")

if __name__ == "__main__":
    main()