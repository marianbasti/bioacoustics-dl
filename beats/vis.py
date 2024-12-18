import matplotlib
matplotlib.use('Agg')

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
from pathlib import Path
import logging
import torchaudio

try:
    import cupy as cp
    from cuml.manifold import TSNE as cuTSNE
    from cuml.manifold import UMAP as cuUMAP
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

from torch.utils.data import DataLoader
from BEATs import BEATs, BEATsConfig
from dataset import AudioDataset
import argparse
import json

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def load_trained_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    cfg = BEATsConfig(checkpoint['cfg'])
    model = BEATs(cfg)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model

def get_audio_metadata(filepath):
    """Extract metadata from audio file."""
    try:
        # Split path and segment info
        actual_path = filepath.split(':')[0]
        info = torchaudio.info(actual_path)
        metadata = {
            'sample_rate': info.sample_rate,
            'num_frames': info.num_frames,
            'num_channels': info.num_channels,
            'duration': f"{info.num_frames / info.sample_rate:.2f}s",
            'filename': Path(actual_path).name
        }
        return metadata
    except Exception as e:
        logging.warning(f"Could not extract metadata for {filepath}: {str(e)}")
        return {'filename': Path(filepath.split(':')[0]).name, 'error': str(e)}

def extract_features(model, dataloader, device, device_ids=None):
    features_list = []
    paths_list = []  # New list to store paths
    metadata_list = []  # New list for metadata
    total_batches = len(dataloader)
    
    if device_ids and len(device_ids) > 1:
        # Multi-GPU case
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    
    # Move model to device(s) once
    model = model.to(device)
    
    # Pin memory in dataloader for faster transfers
    dataloader.pin_memory = True
    
    with torch.cuda.stream(torch.cuda.Stream()):  # Create dedicated CUDA stream
        for batch_idx, (waveform, paths) in enumerate(dataloader):
            logging.info(f"Processing batch {batch_idx+1}/{total_batches}")
            
            # Move data to GPU and ensure contiguous memory
            waveform = waveform.to(device, non_blocking=True).contiguous()
            
            with torch.no_grad():
                # Use model.module.extract_features if DataParallel, otherwise use model.extract_features
                if isinstance(model, torch.nn.DataParallel):
                    features, _ = model.module.extract_features(waveform, padding_mask=None)
                else:
                    features, _ = model.extract_features(waveform, padding_mask=None)
                # Move features to CPU immediately to free GPU memory
                features = features.cpu()
                if len(features.shape) > 2:
                    features = features.mean(dim=1)
                features_list.append(features)
                paths_list.extend(paths)  # Store paths for this batch
                metadata_list.extend([get_audio_metadata(p) for p in paths])
            
            # Log memory usage
            if torch.cuda.is_available():
                logging.info(f"GPU memory: {torch.cuda.memory_allocated(device) / 1024**2:.2f}MB")

    # Concatenate all features on CPU
    final_features = torch.cat(features_list, dim=0)
    logging.info(f"Final combined features shape: {final_features.shape}")
    return final_features, paths_list, metadata_list  # Return features, paths, and metadata

def reduce_dimensions(features, method='tsne', perplexity=30, n_neighbors=15, min_dist=0.1):
    """Reduce feature dimensions using t-SNE or UMAP.
    Returns:
        tuple: (embedded_features, params_str)
    """
    n_samples = features.shape[0]
    
    # Convert features to numpy if they're torch tensors
    if isinstance(features, torch.Tensor):
        features = features.numpy()

    # Standardize features
    features = (features - features.mean(0)) / features.std(0)

    if method == 'tsne':
        # Validate perplexity
        max_perplexity = (n_samples - 1) / 3
        if perplexity > max_perplexity:
            logging.warning(f"Perplexity {perplexity} too large for dataset size {n_samples}. "
                          f"Reducing to {max_perplexity}")
            perplexity = max_perplexity

        try:
            learning_rate = max(200, n_samples / 12)
            
            if GPU_AVAILABLE:
                features_gpu = cp.asarray(features)
                reducer = cuTSNE(n_components=2, 
                               random_state=42, 
                               perplexity=perplexity,
                               learning_rate=learning_rate,
                               init='random')
                embedded = reducer.fit_transform(features_gpu)
                embedded = cp.asnumpy(embedded)
            else:
                reducer = TSNE(n_components=2, 
                             random_state=42, 
                             perplexity=perplexity,
                             learning_rate='auto',
                             init='random')
                embedded = reducer.fit_transform(features)
            params_str = f'perplexity={perplexity}'
            
        except Exception as e:
            logging.error(f"Error during t-SNE: {str(e)}")
            raise
    else:  # umap
        if GPU_AVAILABLE:
            features_gpu = cp.asarray(features)
            reducer = cuUMAP(random_state=42, n_neighbors=n_neighbors, min_dist=min_dist)
            embedded = reducer.fit_transform(features_gpu)
            embedded = cp.asnumpy(embedded)
        else:
            reducer = umap.UMAP(random_state=42, n_neighbors=n_neighbors, min_dist=min_dist)
            embedded = reducer.fit_transform(features)
        params_str = f'n_neighbors={n_neighbors}, min_dist={min_dist}'

    return embedded, params_str

def plot_embedding(embedded, paths, save_path, method, params_str):
    """Plot the embedded features and save to file."""
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(embedded[:, 0], embedded[:, 1], 
                         alpha=0.5, 
                         s=50,
                         c=range(len(embedded)),
                         cmap='viridis')
    
    # Add tooltip functionality
    annot = plt.annotate("", xy=(0,0), xytext=(20,20), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def hover(event):
        if event.inaxes == plt.gca():
            cont, ind = scatter.contains(event)
            if cont:
                pos = scatter.get_offsets()[ind["ind"][0]]
                annot.xy = pos
                text = paths[ind["ind"][0]]
                annot.set_text(text)
                annot.set_visible(True)
                plt.draw()
            else:
                annot.set_visible(False)
                plt.draw()

    plt.gcf().canvas.mpl_connect("motion_notify_event", hover)
    
    plt.colorbar(scatter, label='Sample index')
    plt.title(f'Audio Features Visualization\n{method.upper()}{" (GPU)" if GPU_AVAILABLE else ""}\n({params_str})')
    
    # More descriptive axis labels based on the dimensionality reduction method
    if method.lower() == 'tsne':
        plt.xlabel('t-SNE Projection Component 1')
        plt.ylabel('t-SNE Projection Component 2')
    else:  # umap
        plt.xlabel('UMAP Projection Component 1')
        plt.ylabel('UMAP Projection Component 2')
    
    plt.grid(True, alpha=0.3)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_features(features, save_path, method='tsne', perplexity=30, n_neighbors=15, min_dist=0.1):
    """Legacy wrapper for backward compatibility."""
    embedded, params_str = reduce_dimensions(features, method, perplexity, n_neighbors, min_dist)
    plot_embedding(embedded, save_path, method, params_str)

def parse_grid_params(grid_json):
    """Parse grid search parameters from JSON string."""
    params = json.loads(grid_json)
    grid_configs = []
    
    if 'tsne' in params:
        for perplexity in params['tsne'].get('perplexity', [30]):
            grid_configs.append(('tsne', {'perplexity': perplexity}))
    
    if 'umap' in params:
        umap_params = params['umap']
        for n_neighbors in umap_params.get('n_neighbors', [15]):
            for min_dist in umap_params.get('min_dist', [0.1]):
                grid_configs.append(('umap', {
                    'n_neighbors': n_neighbors,
                    'min_dist': min_dist
                }))
    
    return grid_configs

DEFAULT_GRID = {
    "tsne": {
        "perplexity": [5, 15, 30, 50]
    },
    "umap": {
        "n_neighbors": [5, 15, 30],
        "min_dist": [0.1, 0.3, 0.5]
    }
}

def prepare_features(dataset, checkpoint_path, batch_size=32, device='cuda', device_ids=None):
    """Helper function to prepare features for visualization"""
    model = load_trained_model(checkpoint_path)
    dataloader = DataLoader(dataset, 
                          batch_size=batch_size, 
                          shuffle=False,
                          pin_memory=True,
                          num_workers=2)
    logging.info(f"Found {len(dataset)} audio files")
    features, paths, metadata = extract_features(model, dataloader, device, device_ids)
    return features, paths, metadata

def main():
    setup_logging()
    parser = argparse.ArgumentParser(description='Visualize audio features using BEATs model')
    # Data and model parameters
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Directory containing the audio files to process')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                      help='Path to the trained BEATs model checkpoint (.pt file)')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Number of audio files to process simultaneously. Larger values use more memory but process faster')
    parser.add_argument('--output_dir', type=str, default='feature_visualizations',
                      help='Directory where the visualization plots will be saved')
    
    # Visualization parameters
    parser.add_argument('--methods', nargs='+', choices=['tsne', 'umap'], default=['tsne', 'umap'],
                      help='Dimensionality reduction methods to use. t-SNE is better for local structure, '
                           'UMAP better preserves both local and global structure')
    parser.add_argument('--perplexity', type=float, default=30,
                      help='t-SNE perplexity parameter (5-50). Controls the balance between local and global aspects. '
                           'Larger values consider more neighbors, creating more clustered visualizations')
    parser.add_argument('--n_neighbors', type=int, default=15,
                      help='UMAP neighbors parameter (2-100). Higher values create more global structure. '
                           'Lower values preserve more local structure')
    parser.add_argument('--min_dist', type=float, default=0.1,
                      help='UMAP minimum distance parameter (0.0-1.0). Controls how tightly points cluster. '
                           'Smaller values create tighter clusters')
    
    # Add grid search parameter
    parser.add_argument('--grid_search', type=str, default=json.dumps(DEFAULT_GRID),
                      help='JSON string defining parameter grid. Default explores multiple parameters: '
                           'TSNE perplexity=[5,15,30,50], '
                           'UMAP n_neighbors=[5,15,30] and min_dist=[0.1,0.3,0.5]. '
                           'Example custom grid: '
                           '\'{"tsne": {"perplexity": [10, 30, 50]}, '
                           '"umap": {"n_neighbors": [5, 15, 30], "min_dist": [0.1, 0.5]}}\'')
    
    # Add GPU devices parameter
    parser.add_argument('--gpu_devices', type=str, default='0',
                      help='Comma-separated list of GPU devices to use (e.g., "0,1,2,3"). Default: "0"')
    
    args = parser.parse_args()
    
    # Parse GPU devices
    if torch.cuda.is_available():
        gpu_ids = [int(x) for x in args.gpu_devices.split(',')]
        device = torch.device(f'cuda:{gpu_ids[0]}')
        device_ids = gpu_ids
        logging.info(f"Using devices: {device_ids}")
    else:
        device = torch.device('cpu')
        device_ids = None
        logging.info("Using CPU")
    
    # Load model (will be replicated to each GPU in extract_features)
    model = load_trained_model(args.checkpoint_path)
    dataset = AudioDataset(args.data_dir)
    dataloader = DataLoader(dataset, 
                          batch_size=args.batch_size, 
                          shuffle=False,
                          pin_memory=True,
                          num_workers=2)
    logging.info(f"Found {len(dataset)} audio files")

    # Extract features using multiple GPUs
    features, paths = extract_features(model, dataloader, device, device_ids)
    
    # Create visualizations
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if args.grid_search:
        # Grid search mode
        grid_configs = parse_grid_params(args.grid_search)
        for method, params in grid_configs:
            param_str = '_'.join(f"{k}_{v}" for k, v in params.items())
            save_path = output_dir / f"{method}_{param_str}.png"
            
            if method == 'tsne':
                embedded, params_str = reduce_dimensions(features, method='tsne', perplexity=params['perplexity'])
                plot_embedding(embedded, paths, save_path, method, params_str)
            else:  # umap
                embedded, params_str = reduce_dimensions(features, method='umap',
                                                      n_neighbors=params['n_neighbors'],
                                                      min_dist=params['min_dist'])
                plot_embedding(embedded, paths, save_path, method, params_str)
    else:
        # Original single parameter mode
        for method in args.methods:
            save_path = output_dir / f"{method}_visualization.png"
            visualize_features(
                features, 
                save_path, 
                method=method,
                perplexity=args.perplexity,
                n_neighbors=args.n_neighbors,
                min_dist=args.min_dist
            )

if __name__ == "__main__":
    main()