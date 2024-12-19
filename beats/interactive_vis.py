import streamlit as st
import torch
import plotly.express as px
import pandas as pd
import numpy as np
import argparse
import re
from datetime import datetime
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from dataset import AudioDataset
from vis import extract_features, load_trained_model, prepare_features, reduce_dimensions
from torch.utils.data import DataLoader
import logging
import time
import os

os.environ["NUMBA_CACHE_DIR"] = "/tmp/numba_cache"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@st.cache_resource
def get_args():
    """Cache command line arguments"""
    return parse_args()

def parse_args():
    parser = argparse.ArgumentParser(description='Interactive BEATs Feature Visualization')
    parser.add_argument('--data_dir', type=str, default="path/to/audio/files")
    parser.add_argument('--checkpoint_path', type=str, default="path/to/checkpoint.pt")
    parser.add_argument('--max_samples', type=int, default=None)
    return parser.parse_args()

def analyze_features(features):
    pca = PCA()
    scaled_features = StandardScaler().fit_transform(features)
    pca_result = pca.fit_transform(scaled_features)
    return pca_result, pca.explained_variance_ratio_, np.cumsum(pca.explained_variance_ratio_)

@st.cache_data
def parse_training_log(checkpoint_path):
    log_path = os.path.join(os.path.dirname(checkpoint_path), 'training.log')
    if not os.path.exists(log_path): return None
    
    data = {'epoch': [], 'batch': [], 'loss': []}
    with open(log_path, 'r') as f:
        for line in f:
            if 'Epoch' in line and 'Loss:' in line:
                matches = {
                    'epoch': re.search(r'Epoch (\d+)', line),
                    'batch': re.search(r'Batch (\d+)', line),
                    'loss': re.search(r'Loss: ([\d.]+)', line)
                }
                if all(matches.values()):
                    data['epoch'].append(int(matches['epoch'].group(1)))
                    data['batch'].append(int(matches['batch'].group(1)))
                    data['loss'].append(float(matches['loss'].group(1)))
    
    return pd.DataFrame(data)

def create_feature_analysis_tab(features, paths, metadata):
    """Create additional analysis visualizations"""
    st.header("Feature Analysis")
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(["Clustering", "PCA Analysis", "Temporal Patterns", "Training Loss"])
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

    with tab4:
        st.markdown("""
        **Training Loss**
        
        This graph shows the evolution of the training loss over time. Lower values indicate better model performance.
        The x-axis represents training progress through epochs and batches, while the y-axis shows the loss value.
        """)
        
        loss_df = parse_training_log(st.session_state.args.checkpoint_path)
        if loss_df is not None:
            # Create global step for x-axis
            loss_df['step'] = loss_df['epoch'] * loss_df['batch'].max() + loss_df['batch']
            
            fig_loss = px.line(loss_df, x='step', y='loss', 
                             title='Training Loss Over Time',
                             labels={'step': 'Training Steps', 'loss': 'Loss Value'})
            
            # Add epoch markers
            epoch_changes = loss_df[loss_df['batch'] == 1]
            for _, row in epoch_changes.iterrows():
                fig_loss.add_vline(x=row['step'], line_dash="dash", 
                                 annotation_text=f"Epoch {row['epoch']}")
            
            st.plotly_chart(fig_loss)
        else:
            st.warning("No training log file found at the checkpoint location.")

@st.cache_data
def load_features(data_dir, checkpoint_path, batch_size, max_samples=None):
    start_time = time.time()
    dataset = AudioDataset(
        root_dir=data_dir,
        segment_duration=10,
        overlap=0.0,
        max_segments_per_file=5,
        random_segments=True,
        max_samples=max_samples
    )
    
    features, paths, metadata = prepare_features(dataset, checkpoint_path, batch_size, device)
    if isinstance(features, torch.Tensor):
        features = features.numpy()
    
    logger.info(f"Features loaded in {time.time() - start_time:.2f} seconds")
    return features, paths, metadata

def extract_date(filename):
    match = re.search(r'(\d{8})', filename)
    return datetime.strptime(match.group(1), '%Y%m%d') if match else None

def get_seasonal_color_value(date):
    return ((date.timetuple().tm_yday + 9) % 365) / 365.0

def create_seasonal_colorscale():
    return [
        [0.0, 'rgb(255,0,0)'],     # summer
        [0.25, 'rgb(255,165,0)'],  # autumn
        [0.5, 'rgb(0,25,255)'],    # winter
        [0.75, 'rgb(15,255,15)'],  # spring
        [1.0, 'rgb(255,0,0)']      # summer
    ]

def reduce_dimensions_3d(features, method, **params):
    if method == 'tsne':
        from sklearn.manifold import TSNE
        reducer_2d = TSNE(n_components=2, **params)
        reducer_1d = TSNE(n_components=1, **params)
    elif method == 'umap':
        import umap
        reducer_2d = umap.UMAP(n_components=2, **params)
        reducer_1d = umap.UMAP(n_components=1, **params)
    
    embedded_2d = reducer_2d.fit_transform(features)
    third_dim = reducer_1d.fit_transform(features).flatten()
    
    if 'reducers' not in st.session_state:
        st.session_state.reducers = {}
    st.session_state.reducers[f'{method}_2d'] = reducer_2d
    st.session_state.reducers[f'{method}_1d'] = reducer_1d
    
    embedded_3d = np.column_stack((embedded_2d, third_dim))
    params_str = f"{method.upper()}-3D"
    
    return embedded_3d, params_str

def create_plot(embedded, paths, metadata, method, params_str, point_size):
    df = pd.DataFrame({
        f'{method}_1': embedded[:, 0],
        f'{method}_2': embedded[:, 1],
        f'{method}_3': embedded[:, 2],
    })
    
    for key in metadata[0].keys():
        df[key] = [m[key] for m in metadata]
    
    df['date'] = [extract_date(path) for path in paths]
    df['seasonal_value'] = df['date'].apply(get_seasonal_color_value)
    
    hover_template = (
        "<b>%{customdata[0]}</b><br>"
        "Date: %{customdata[4]}<br>"
        "Sample Rate: %{customdata[1]} Hz<br>"
        "Duration: %{customdata[2]}<br>"
        "Channels: %{customdata[3]}<br>"
        "Feature Depth: %{customdata[5]:.2f}"
    )
    
    fig = go.Figure(data=[go.Scatter3d(
        x=df[f'{method}_1'],
        y=df[f'{method}_2'],
        z=df[f'{method}_3'],
        mode='markers',
        marker=dict(
            size=point_size,
            color=df['seasonal_value'],
            colorscale=create_seasonal_colorscale(),
            opacity=0.8
        ),
        customdata=np.column_stack((
            df['filename'], 
            df['sample_rate'], 
            df['duration'], 
            df['num_channels'],
            df['date'].dt.strftime('%Y-%m-%d'),
            df[f'{method}_3']
        )),
        hovertemplate=hover_template
    )])
    
    fig.update_layout(
        title=f'{method.upper()} 3D Visualization ({params_str})',
        scene=dict(
            xaxis_title=f'{method}_1',
            yaxis_title=f'{method}_2',
            zaxis_title=f'{method}_3',
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=700,
        height=700,
    )
    
    return fig

def update_plot_with_new_point(fig, new_point_coords, point_size):
    new_trace = go.Scatter3d(
        x=[new_point_coords[0]],
        y=[new_point_coords[1]],
        z=[new_point_coords[2]],
        mode='markers',
        marker=dict(
            color='white',
            size=point_size * 2,
            line=dict(
                color='red',
                width=2
            ),
            symbol='circle'
        ),
        name=f'New Point {len(fig.data)}',
        showlegend=True
    )
    
    fig.add_trace(new_trace)
    return fig

def add_point_to_embedding(existing_features, new_features, existing_embedding, method, **params):
    if method == 'tsne':
        combined_features = np.vstack([existing_features, new_features])
        combined_embedded, _ = reduce_dimensions_3d(combined_features, method=method, **params)
        return combined_embedded[-len(new_features):]
    elif method == 'umap':
        reducer_2d = st.session_state.reducers[f'{method}_2d']
        reducer_1d = st.session_state.reducers[f'{method}_1d']
        
        new_2d = reducer_2d.transform(new_features)
        new_1d = reducer_1d.transform(new_features).flatten()
        
        return np.column_stack((new_2d, new_1d))

def process_multiple_files(uploaded_files, model, device):
    temp_dir = "./temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        for file in uploaded_files:
            with open(os.path.join(temp_dir, file.name), "wb") as f:
                f.write(file.getvalue())
        
        dataset = AudioDataset(root_dir=temp_dir, max_segments_per_file=1)
        dataloader = DataLoader(dataset, batch_size=len(uploaded_files), shuffle=False)
        features, _, metadata = extract_features(model, dataloader, device)
        return features.cpu().numpy(), metadata
    
    finally:
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def main():
    if 'initialized' not in st.session_state:
        st.session_state.update({
            'initialized': True,
            'args': get_args(),
            'new_tsne_points': [],
            'new_umap_points': []
        })
    
    logger = logging.getLogger(__name__)
    st.set_page_config(layout="wide", page_title="BEATs Feature Visualization")
    st.title("Interactive BEATs Feature Visualization")

    with st.sidebar:
        st.header("Parameters")
        data_dir = st.text_input(
            "Data Directory", 
            value=st.session_state.args.data_dir,
            help="Directory containing the audio files to analyze"
        )
        checkpoint_path = st.text_input(
            "Checkpoint Path", 
            value=st.session_state.args.checkpoint_path,
            help="Path to the pre-trained BEATs model checkpoint file (.pt)"
        )
        batch_size = st.number_input(
            "Batch Size", 
            min_value=1, 
            value=32,
            help="Number of samples processed together. Higher values use more memory but may be faster"
        )
        
        st.divider()
        st.header("t-SNE Parameters")
        perplexity = st.slider(
            "Perplexity", 
            min_value=5, 
            max_value=100, 
            value=30,
            help="Balance between preserving local and global structure. Lower values focus on local structure, higher values on global"
        )
        
        st.divider()
        st.header("UMAP Parameters")
        n_neighbors = st.slider(
            "Number of Neighbors", 
            min_value=2, 
            max_value=100, 
            value=15,
            help="Controls how UMAP balances local versus global structure. Higher values result in more global structure being preserved"
        )
        min_dist = st.slider(
            "Minimum Distance", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.1, 
            step=0.05,
            help="Controls how tightly UMAP packs points together. Lower values create tighter clusters"
        )

        st.divider()
        st.header("Visualization Parameters")
        point_size = st.slider(
            "Point Size",
            min_value=1,
            max_value=12,
            value=2,
            help="Size of the points in the scatter plots"
        )

    try:
        if 'features_loaded' not in st.session_state:
            logger.info("Attempting to load features...")
            features, paths, metadata = load_features(data_dir, checkpoint_path, batch_size, max_samples=st.session_state.args.max_samples)
            st.session_state.features_loaded = True
        else:
            features, paths, metadata = load_features(data_dir, checkpoint_path, batch_size, max_samples=st.session_state.args.max_samples)
    except Exception as e:
        logger.error(f"Error loading features: {str(e)}", exc_info=True)
        st.error(f"Error loading features: {str(e)}")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.header("t-SNE Visualization")
        try:
            start_time = time.time()
            tsne_embedded, tsne_params = reduce_dimensions_3d(
                features, 
                method='tsne',
                perplexity=perplexity
            )
            st.session_state.tsne_embedded = tsne_embedded
            logger.info(f"t-SNE completed in {time.time() - start_time:.2f} seconds")
            fig_tsne = create_plot(tsne_embedded, paths, metadata, 't-SNE', tsne_params, point_size)
            st.session_state.fig_tsne = fig_tsne
            st.plotly_chart(fig_tsne)
        except Exception as e:
            logger.error(f"Error in t-SNE: {str(e)}", exc_info=True)
            st.error(f"Error in t-SNE: {str(e)}")

    with col2:
        st.header("UMAP Visualization")
        try:
            start_time = time.time()
            umap_embedded, umap_params = reduce_dimensions_3d(
                features,
                method='umap',
                n_neighbors=n_neighbors,
                min_dist=min_dist
            )
            st.session_state.umap_embedded = umap_embedded
            logger.info(f"UMAP completed in {time.time() - start_time:.2f} seconds")
            fig_umap = create_plot(umap_embedded, paths, metadata, 'UMAP', umap_params, point_size)
            st.session_state.fig_umap = fig_umap
            st.plotly_chart(fig_umap)
        except Exception as e:
            logger.error(f"Error in UMAP: {str(e)}", exc_info=True)
            st.error(f"Error in UMAP: {str(e)}")

    st.markdown("### Drag and Drop Audio Files")
    uploaded_files = st.file_uploader(
        "Drop audio files here to see where they appear in the feature space",
        accept_multiple_files=True,
        type=['wav', 'mp3', 'ogg', 'flac']
    )

    if uploaded_files:
        logger.info(f"Processing {len(uploaded_files)} uploaded files")
        model = load_trained_model(checkpoint_path)
        model.to(device)
        model.eval()
        
        col1, col2 = st.columns(2)
        
        try:
            with st.spinner('Processing uploaded files...'):
                new_features, new_metadata = process_multiple_files(uploaded_files, model, device)
                
                with col1:
                    new_tsne_points = add_point_to_embedding(
                        features,
                        new_features,
                        st.session_state.tsne_embedded,
                        'tsne',
                        perplexity=perplexity
                    )
                    st.session_state.new_tsne_points.extend(new_tsne_points)
                    updated_tsne_fig = st.session_state.fig_tsne
                    for point in new_tsne_points:
                        updated_tsne_fig = update_plot_with_new_point(
                            updated_tsne_fig,
                            point,
                            point_size
                        )
                    st.session_state.fig_tsne = updated_tsne_fig
                    st.plotly_chart(updated_tsne_fig, use_container_width=True)
                
                with col2:
                    new_umap_points = add_point_to_embedding(
                        features,
                        new_features,
                        st.session_state.umap_embedded,
                        'umap',
                        n_neighbors=n_neighbors,
                        min_dist=min_dist
                    )
                    st.session_state.new_umap_points.extend(new_umap_points)
                    updated_umap_fig = st.session_state.fig_umap
                    for point in new_umap_points:
                        updated_umap_fig = update_plot_with_new_point(
                            updated_umap_fig,
                            point,
                            point_size
                        )
                    st.session_state.fig_umap = updated_umap_fig
                    st.plotly_chart(updated_umap_fig, use_container_width=True)
            
            st.success(f"Successfully processed {len(uploaded_files)} files")
            
        except Exception as e:
            logger.error(f"Error processing files: {str(e)}", exc_info=True)
            st.error(f"Error processing files: {str(e)}")

    st.divider()
    create_feature_analysis_tab(features, paths, metadata)
    logger.info("Application rendering completed")

if __name__ == "__main__":
    main()