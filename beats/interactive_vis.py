import streamlit as st
import torch
import plotly.express as px
import pandas as pd
from vis import prepare_features, reduce_dimensions
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Interactive BEATs Feature Visualization')
    parser.add_argument('--data_dir', type=str, default="path/to/audio/files",
                      help='Directory containing the audio files')
    parser.add_argument('--checkpoint_path', type=str, default="path/to/checkpoint.pt",
                      help='Path to the trained BEATs model checkpoint')
    return parser.parse_args()

@st.cache_data
def load_features(data_dir, checkpoint_path, batch_size):
    """Cache the feature extraction to avoid recomputing"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    features, paths, metadata = prepare_features(data_dir, checkpoint_path, batch_size, device)
    if isinstance(features, torch.Tensor):
        features = features.numpy()
    return features, paths, metadata

def create_plot(embedded, paths, metadata, method, params_str):
    """Create an interactive scatter plot using plotly"""
    df = pd.DataFrame({
        f'{method}_1': embedded[:, 0],
        f'{method}_2': embedded[:, 1],
        'index': np.arange(len(embedded))
    })
    
    # Add metadata columns
    for key in metadata[0].keys():
        df[key] = [m[key] for m in metadata]
    
    # Customize hover template
    hover_template = (
        "<b>%{customdata[0]}</b><br>"
        "Sample Rate: %{customdata[1]} Hz<br>"
        "Duration: %{customdata[2]}<br>"
        "Channels: %{customdata[3]}"
    )
    
    fig = px.scatter(
        df,
        x=f'{method}_1',
        y=f'{method}_2',
        title=f'{method.upper()} Visualization ({params_str})',
        color='index',
        color_continuous_scale='viridis',
        custom_data=['filename', 'sample_rate', 'duration', 'num_channels']
    )
    
    fig.update_traces(
        hovertemplate=hover_template,
        marker=dict(size=8)
    )
    
    fig.update_layout(
        width=600,
        height=600,
        coloraxis_colorbar_title='Sample Index'
    )
    
    return fig

def main():
    # Parse command line arguments
    args = parse_args()
    
    st.set_page_config(layout="wide", page_title="BEATs Feature Visualization")
    st.title("Interactive Audio Feature Visualization")

    # Sidebar for input parameters
    with st.sidebar:
        st.header("Parameters")
        data_dir = st.text_input("Data Directory", value=args.data_dir)
        checkpoint_path = st.text_input("Checkpoint Path", value=args.checkpoint_path)
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
        features, paths, metadata = load_features(data_dir, checkpoint_path, batch_size)
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
            fig_tsne = create_plot(tsne_embedded, paths, metadata, 't-SNE', tsne_params)
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
            fig_umap = create_plot(umap_embedded, paths, metadata, 'UMAP', umap_params)
            st.plotly_chart(fig_umap)
        except Exception as e:
            st.error(f"Error in UMAP: {str(e)}")

if __name__ == "__main__":
    main()