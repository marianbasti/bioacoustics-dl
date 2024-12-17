import streamlit as st
import torch
import plotly.express as px
import pandas as pd
from vis import prepare_features, reduce_dimensions
import numpy as np
import argparse
import re
from datetime import datetime
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from dataset import AudioDataset
from vis import extract_features, load_trained_model
from torch.utils.data import DataLoader

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

def parse_args():
    parser = argparse.ArgumentParser(description='Interactive BEATs Feature Visualization')
    parser.add_argument('--data_dir', type=str, default="path/to/audio/files",
                      help='Directory containing the audio files')
    parser.add_argument('--checkpoint_path', type=str, default="path/to/checkpoint.pt",
                      help='Path to the trained BEATs model checkpoint')
    return parser.parse_args()

def analyze_features(features):
    """Perform statistical analysis of features"""
    # PCA Analysis
    pca = PCA()
    scaled_features = StandardScaler().fit_transform(features)
    pca_result = pca.fit_transform(scaled_features)
    
    # Explained variance
    exp_var_ratio = pca.explained_variance_ratio_
    cum_sum_eigenvalues = np.cumsum(exp_var_ratio)
    
    return pca_result, exp_var_ratio, cum_sum_eigenvalues

def create_feature_analysis_tab(features, paths, metadata):
    """Create additional analysis visualizations"""
    st.header("Feature Analysis")
    
    # Create tabs for different analyses
    tab1, tab2, tab3 = st.tabs(["Clustering", "PCA Analysis", "Temporal Patterns"])
    # PCA Analysis
    pca_result, exp_var_ratio, cum_sum = analyze_features(features)
    
    with tab1:
        st.markdown("""
        **K-means Clustering**
        
        El algoritmo K-means es una técnica de agrupamiento que divide los datos en grupos (clusters) 
        basándose en la similitud de sus características. Cada punto se asigna al grupo cuyo centro 
        (centroide) está más cerca. Esto nos permite identificar patrones y grupos naturales en 
        nuestros datos de audio.
        """)
        # K-means clustering
        n_clusters = st.slider("Number of Clusters", 2, 10, 4)
        kmeans = KMeans(n_clusters=n_clusters)
        clusters = kmeans.fit_predict(features)
        
        # Plot with clusters
        df_clusters = pd.DataFrame({
            'PC1': pca_result[:, 0],
            'PC2': pca_result[:, 1],
            'Cluster': clusters
        })
        
        fig_clusters = px.scatter(df_clusters, x='PC1', y='PC2', 
                                color='Cluster', title='Feature Clusters')
        st.plotly_chart(fig_clusters)
    
    with tab2:
        st.markdown("""
        **Análisis de Componentes Principales (PCA)**
        
        PCA es una técnica que reduce la dimensionalidad de los datos, manteniendo la mayor cantidad 
        de información posible. El gráfico muestra cuánta varianza explica cada componente principal 
        (barras azules) y la varianza acumulada (línea naranja). Esto nos ayuda a entender la 
        complejidad y estructura de nuestros datos de audio.
        """)
        # Scree plot
        x_values = list(range(1, len(exp_var_ratio) + 1))  # Convert range to list
        fig_pca = go.Figure(data=[
            go.Bar(name='Individual', x=x_values, y=exp_var_ratio),
            go.Scatter(name='Cumulative', x=x_values, y=cum_sum, yaxis='y2')
        ])
        
        fig_pca.update_layout(
            title='PCA Explained Variance',
            yaxis=dict(title='Explained Variance Ratio'),
            yaxis2=dict(title='Cumulative Explained Variance',
                       overlaying='y', side='right')
        )
        st.plotly_chart(fig_pca)
        
    with tab3:
        st.markdown("""
        **Patrones Temporales**
        
        Esta visualización muestra la distribución temporal de las grabaciones de audio a lo largo 
        de los meses del año. Nos permite identificar patrones estacionales y temporales en la 
        recolección de datos, lo cual puede ser útil para entender comportamientos acústicos 
        específicos de ciertas épocas del año.
        """)
        # Temporal patterns
        dates = [extract_date(p) for p in paths]
        months = [d.month for d in dates if d]
        
        fig_temporal = px.histogram(x=months, nbins=12,
                                  title='Monthly Distribution',
                                  labels={'x': 'Month', 'y': 'Count'})
        st.plotly_chart(fig_temporal)

@st.cache_data
def load_features(data_dir, checkpoint_path, batch_size):
    """Cache the feature extraction to avoid recomputing"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    features, paths, metadata = prepare_features(data_dir, checkpoint_path, batch_size, device)
    if isinstance(features, torch.Tensor):
        features = features.numpy()
    return features, paths, metadata

def extract_date(filename):
    """Extract date from filename using regex"""
    match = re.search(r'(\d{8})', filename)
    if match:
        return datetime.strptime(match.group(1), '%Y%m%d')
    return None

def get_seasonal_color_value(date):
    """Convert date to seasonal value (0-1) for Southern Hemisphere"""
    day_of_year = date.timetuple().tm_yday
    shifted_day = (day_of_year+9) % 365
    return shifted_day / 365.0

def create_seasonal_colorscale():
    """Create a custom colorscale for seasons in Southern Hemisphere"""
    return [
        [0.0, 'rgb(255,0,0)'],     # red (summer - December 21)
        [0.25, 'rgb(255,165,0)'],   # orange (autumn - April 21)
        [0.5, 'rgb(0,25,255)'],   # blue (winter - July 21)
        [0.75, 'rgb(15,255,15)'],   # green (spring - September 21)
        [1.0, 'rgb(255,0,0)'],     # back to red (summer)
    ]

def create_plot(embedded, paths, metadata, method, params_str):
    """Create an interactive scatter plot using plotly"""
    df = pd.DataFrame({
        f'{method}_1': embedded[:, 0],
        f'{method}_2': embedded[:, 1],
    })
    
    # Add metadata columns and extract dates
    for key in metadata[0].keys():
        df[key] = [m[key] for m in metadata]
    
    # Extract dates and seasonal values
    df['date'] = [extract_date(path) for path in paths]
    df['seasonal_value'] = df['date'].apply(get_seasonal_color_value)
    
    # Customize hover template
    hover_template = (
        "<b>%{customdata[0]}</b><br>"
        "Date: %{customdata[4]}<br>"
        "Sample Rate: %{customdata[1]} Hz<br>"
        "Duration: %{customdata[2]}<br>"
        "Channels: %{customdata[3]}"
    )
    
    fig = px.scatter(
        df,
        x=f'{method}_1',
        y=f'{method}_2',
        title=f'{method.upper()} Visualization ({params_str})',
        color='seasonal_value',
        color_continuous_scale=create_seasonal_colorscale(),
        custom_data=['filename', 'sample_rate', 'duration', 'num_channels', 
                    df['date'].dt.strftime('%Y-%m-%d')]
    )
    
    fig.update_traces(
        hovertemplate=hover_template,
        marker=dict(size=8)
    )
    
    fig.update_layout(
        width=600,
        height=600,
        coloraxis_colorbar_title='Season'
    )
    
    return fig

def process_dropped_file(uploaded_file, model, device):
    """Process a single uploaded audio file and extract features"""
    # Save uploaded file temporarily
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    
    # Create temporary dataset with single file
    dataset = AudioDataset(root_dir=".", file_list=[temp_path])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Extract features
    features, _, metadata = extract_features(model, dataloader, device)
    
    # Cleanup
    import os
    os.remove(temp_path)
    
    return features.numpy(), metadata[0]

def update_plot_with_new_point(fig, new_point_coords):
    """Add a new white point to the existing plot"""
    fig.add_trace(
        go.Scatter(
            x=[new_point_coords[0]],
            y=[new_point_coords[1]],
            mode='markers',
            marker=dict(
                color='white',
                size=12,
                line=dict(
                    color='black',
                    width=2
                )
            ),
            name='New Point',
            showlegend=True
        )
    )
    return fig

def main():
    # Parse command line arguments
    args = parse_args()
    
    st.set_page_config(layout="wide", page_title="BEATs Feature Visualization")
    st.title("Interactive BEATs Feature Visualization")

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

    # Create drag-and-drop area
    st.markdown("### Drag and Drop Audio Files")
    uploaded_files = st.file_uploader(
        "Drop audio files here to see where they appear in the feature space",
        accept_multiple_files=True,
        type=['wav', 'mp3', 'ogg', 'flac']
    )

    # Process uploaded files if any
    if uploaded_files:
        model = load_trained_model(checkpoint_path)
        model.to(device)
        
        for uploaded_file in uploaded_files:
            st.write(f"Processing {uploaded_file.name}...")
            
            # Extract features from new file
            new_features, new_metadata = process_dropped_file(uploaded_file, model, device)
            
            # Reduce dimensions of new point using the same parameters
            if 'tsne_embedded' in locals():
                new_tsne, _ = reduce_dimensions(
                    new_features,
                    method='tsne',
                    perplexity=perplexity
                )
                fig_tsne = update_plot_with_new_point(fig_tsne, new_tsne[0])
                col1.plotly_chart(fig_tsne)
            
            if 'umap_embedded' in locals():
                new_umap, _ = reduce_dimensions(
                    new_features,
                    method='umap',
                    n_neighbors=n_neighbors,
                    min_dist=min_dist
                )
                fig_umap = update_plot_with_new_point(fig_umap, new_umap[0])
                col2.plotly_chart(fig_umap)

    st.divider()
    create_feature_analysis_tab(features, paths, metadata)

if __name__ == "__main__":
    main()