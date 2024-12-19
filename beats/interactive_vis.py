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
import logging
import time
import os

os.environ["NUMBA_CACHE_DIR"] = "/tmp/numba_cache"  # Add at the very top

# Move logging setup outside of cache
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Remove @st.cache_data from setup_logging since we don't need it anymore
def setup_logging():
    return logger

@st.cache_resource
def get_args():
    """Cache command line arguments"""
    return parse_args()

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
    parser.add_argument('--max_samples', type=int, default=None,
                      help='Set a maximum amount of samples to load')
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

@st.cache_data
def parse_training_log(checkpoint_path):
    """Parse training log file to extract loss values"""
    log_path = os.path.join(os.path.dirname(checkpoint_path), 'training.log')
    if not os.path.exists(log_path):
        return None
    
    epochs = []
    batches = []
    losses = []
    
    with open(log_path, 'r') as f:
        for line in f:
            if 'Epoch' in line and 'Loss:' in line:
                try:
                    # Extract values using regex
                    epoch_match = re.search(r'Epoch (\d+)', line)
                    batch_match = re.search(r'Batch (\d+)', line)
                    loss_match = re.search(r'Loss: ([\d.]+)', line)
                    
                    if epoch_match and batch_match and loss_match:
                        epochs.append(int(epoch_match.group(1)))
                        batches.append(int(batch_match.group(1)))
                        losses.append(float(loss_match.group(1)))
                except:
                    continue
    
    return pd.DataFrame({
        'epoch': epochs,
        'batch': batches,
        'loss': losses
    })

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
    """Cache the feature extraction to avoid recomputing"""
    logger.info(f"Loading features from {data_dir}")
    start_time = time.time()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create dataset
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
    
    logger.info(f"Features loaded successfully in {time.time() - start_time:.2f} seconds")
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

def reduce_dimensions_3d(features, method, **params):
    """Reduce dimensions to 3D in one step instead of separate 2D+1D"""
    if method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=3, **params)
    elif method == 'umap':
        import umap
        reducer = umap.UMAP(n_components=3, **params)
    
    embedded = reducer.fit_transform(features)
    
    # Store reducer in session state for reuse
    if 'reducers' not in st.session_state:
        st.session_state.reducers = {}
    st.session_state.reducers[method] = reducer
    
    params_str = f"{method.upper()}-3D"
    return embedded, params_str

def create_plot(embedded, paths, metadata, method, params_str, point_size):
    """Create an interactive 3D scatter plot using plotly"""
    df = pd.DataFrame({
        f'{method}_1': embedded[:, 0],
        f'{method}_2': embedded[:, 1],
        f'{method}_3': embedded[:, 2],  # Now using same algorithm
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
        "Channels: %{customdata[3]}<br>"
        "Feature Depth: %{customdata[5]:.2f}"
    )
    
    fig = go.Figure(data=[go.Scatter3d(
        x=df[f'{method}_1'],
        y=df[f'{method}_2'],
        z=df[f'{method}_3'],  # Updated variable name
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
            df[f'{method}_3']  # Updated variable name
        )),
        hovertemplate=hover_template
    )])
    
    fig.update_layout(
        title=f'{method.upper()} 3D Visualization ({params_str})',
        scene=dict(
            xaxis_title=f'{method}_1',
            yaxis_title=f'{method}_2',
            zaxis_title=f'{method}_3',  # Updated title
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
    """Add a new point to the existing 3D plot"""
    new_trace = go.Scatter3d(
        x=[new_point_coords[0]],
        y=[new_point_coords[1]],
        z=[new_point_coords[2]],
        mode='markers',
        marker=dict(
            color='white',
            size=point_size * 2,  # Make new points more visible
            line=dict(
                color='red',
                width=2
            ),
            symbol='circle'
        ),
        name=f'New Point {len(fig.data)}',  # Unique name for each new point
        showlegend=True
    )
    
    fig.add_trace(new_trace)
    return fig

def add_point_to_embedding(existing_features, new_features, existing_embedding, method, **params):
    """Project new points using the stored transformation"""
    if method == 'tsne':
        # For t-SNE, we need to refit with both old and new points to maintain consistency
        from sklearn.manifold import TSNE
        combined_features = np.vstack([existing_features, new_features])
        reducer = TSNE(n_components=3, **params)
        combined_embedding = reducer.fit_transform(combined_features)
        # Return both full embedding and new points
        return combined_embedding, combined_embedding[-len(new_features):]
    elif method == 'umap':
        # UMAP supports transform, so we can use the stored model
        reducer = st.session_state.reducers[method]
        new_embedding = reducer.transform(new_features)
        return existing_embedding, new_embedding

def process_multiple_files(uploaded_files, model, device):
    """Process multiple uploaded audio files at once and extract features"""
    
    # Create temp directory if it doesn't exist
    temp_dir = "./temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Save all uploaded files temporarily
        for uploaded_file in uploaded_files:
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())
        
        # Create dataset with all uploaded files
        dataset = AudioDataset(
            root_dir=temp_dir,
            max_segments_per_file=1,
        )
        dataloader = DataLoader(dataset, batch_size=len(uploaded_files), shuffle=False)
        
        # Extract features
        features, _, metadata = extract_features(model, dataloader, device)
        
        return features.cpu().numpy(), metadata
    
    finally:
        # Cleanup
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def main():
    logger = setup_logging()
    st.set_page_config(layout="wide", page_title="BEATs Feature Visualization")
    # Use session state to track initialization
    if 'initialized' not in st.session_state:
        logger.info("Starting BEATs Feature Visualization application")
        st.session_state.initialized = True
        st.session_state.args = get_args()
    
    args = st.session_state.args

    # Remove the logging message here since we moved it to first initialization
    st.title("Interactive BEATs Feature Visualization")

    # Sidebar for input parameters
    with st.sidebar:
        st.header("Parameters")
        data_dir = st.text_input(
            "Data Directory", 
            value=args.data_dir,
            help="Directory containing the audio files to analyze"
        )
        checkpoint_path = st.text_input(
            "Checkpoint Path", 
            value=args.checkpoint_path,
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

    # Load features (cached)
    try:
        if 'features_loaded' not in st.session_state:
            logger.info("Attempting to load features...")
            features, paths, metadata = load_features(data_dir, checkpoint_path, batch_size, max_samples=args.max_samples)
            st.session_state.features_loaded = True
        else:
            features, paths, metadata = load_features(data_dir, checkpoint_path, batch_size, max_samples=args.max_samples)
    except Exception as e:
        logger.error(f"Error loading features: {str(e)}", exc_info=True)
        st.error(f"Error loading features: {str(e)}")
        return

    # Create two columns for visualizations
    col1, col2 = st.columns(2)

    with col1:
        st.header("t-SNE Visualization")
        try:
            start_time = time.time()
            tsne_embedded, tsne_params = reduce_dimensions_3d(  # Changed to 3D
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
            umap_embedded, umap_params = reduce_dimensions_3d(  # Changed to 3D
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

    # Initialize lists to store new points if not already in session state
    if 'new_tsne_points' not in st.session_state:
        st.session_state.new_tsne_points = []
    if 'new_umap_points' not in st.session_state:
        st.session_state.new_umap_points = []

    # Create drag-and-drop area
    st.markdown("### Drag and Drop Audio Files")
    uploaded_files = st.file_uploader(
        "Drop audio files here to see where they appear in the feature space",
        accept_multiple_files=True,
        type=['wav', 'mp3', 'ogg', 'flac']
    )

    # Process uploaded files if any
    if uploaded_files:
        logger.info(f"Processing {len(uploaded_files)} uploaded files")
        model = load_trained_model(checkpoint_path)
        model.to(device)
        model.eval()  # Ensure model is in eval mode
        
        col1, col2 = st.columns(2)
        
        try:
            with st.spinner('Processing uploaded files...'):
                # Extract features from all new files at once
                new_features, new_metadata = process_multiple_files(uploaded_files, model, device)
                
                # Update t-SNE plot
                with col1:
                    full_tsne_embedded, new_tsne_points = add_point_to_embedding(
                        features,
                        new_features,
                        st.session_state.tsne_embedded,
                        'tsne',
                        perplexity=perplexity
                    )
                    # Update the full embedding in session state
                    st.session_state.tsne_embedded = full_tsne_embedded
                    # Create new plot with full embedding
                    fig_tsne = create_plot(full_tsne_embedded, paths + [f.name for f in uploaded_files], 
                                         metadata + new_metadata, 't-SNE', 'Updated t-SNE', point_size)
                    st.plotly_chart(fig_tsne)
                
                # Update UMAP plot (similar changes)
                with col2:
                    full_umap_embedded, new_umap_points = add_point_to_embedding(
                        features,
                        new_features,
                        st.session_state.umap_embedded,
                        'umap',
                        n_neighbors=n_neighbors,
                        min_dist=min_dist
                    )
                    st.session_state.umap_embedded = full_umap_embedded
                    fig_umap = create_plot(full_umap_embedded, paths + [f.name for f in uploaded_files],
                                         metadata + new_metadata, 'UMAP', 'Updated UMAP', point_size)
                    st.plotly_chart(fig_umap)
            
            st.success(f"Successfully processed {len(uploaded_files)} files")
            
        except Exception as e:
            logger.error(f"Error processing files: {str(e)}", exc_info=True)
            st.error(f"Error processing files: {str(e)}")

    st.divider()
    create_feature_analysis_tab(features, paths, metadata)
    logger.info("Application rendering completed")

if __name__ == "__main__":
    main()