import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
from pathlib import Path
import logging

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

def extract_features(model, dataloader, devices):
    if not isinstance(devices, list):
        devices = [devices]
    
    # Create model replicas for each GPU
    models = {device: model.to(device) for device in devices}
    features_list = []
    total_batches = len(dataloader)
    
    logging.info(f"Starting feature extraction on devices: {devices}")
    
    for batch_idx, (waveform, paths) in enumerate(dataloader):
        # Distribute batches across GPUs in round-robin fashion
        device = devices[batch_idx % len(devices)]
        current_model = models[device]
        
        # Log progress
        logging.info(f"Batch {batch_idx+1}/{total_batches} on {device}")
        logging.info(f"Input waveform shape: {waveform.shape}")
        
        # Move waveform to current device
        waveform = waveform.to(device)
        
        # Extract features
        with torch.no_grad():
            features, _ = current_model.extract_features(waveform, padding_mask=None)
            logging.info(f"Extracted features shape: {features.shape}")
        
        features_list.append(features.cpu())
    
    final_features = torch.cat(features_list, dim=0)
    logging.info(f"Final combined features shape: {final_features.shape}")

    if len(final_features.shape) > 2:
        final_features = final_features.mean(dim=1)

    return final_features

def visualize_features(features, save_path, method='tsne', perplexity=30, n_neighbors=15, min_dist=0.1):
    # Dimensionality reduction
    if method == 'tsne':
        if GPU_AVAILABLE:
            features_gpu = cp.asarray(features)
            reducer = cuTSNE(n_components=2, random_state=42, perplexity=perplexity)
            embedded = reducer.fit_transform(features_gpu)
            embedded = cp.asnumpy(embedded)
            params_str = f'perplexity={perplexity}'
        else:
            reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            embedded = reducer.fit_transform(features)
            params_str = f'perplexity={perplexity}'
    else:  # umap
        if GPU_AVAILABLE:
            features_gpu = cp.asarray(features)
            reducer = cuUMAP(random_state=42, n_neighbors=n_neighbors, min_dist=min_dist)
            embedded = reducer.fit_transform(features_gpu)
            embedded = cp.asnumpy(embedded)
            params_str = f'n_neighbors={n_neighbors}, min_dist={min_dist}'
        else:
            reducer = umap.UMAP(random_state=42, n_neighbors=n_neighbors, min_dist=min_dist)
            embedded = reducer.fit_transform(features)
            params_str = f'n_neighbors={n_neighbors}, min_dist={min_dist}'
    
    # Create visualization
    plt.figure(figsize=(10, 10))
    plt.scatter(embedded[:, 0], embedded[:, 1], alpha=0.5)
    plt.title(f'Audio Features Visualization\n{method.upper()}{" (GPU)" if GPU_AVAILABLE else ""}\n({params_str})')
    plt.savefig(save_path)
    plt.close()

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
    gpu_ids = [int(x) for x in args.gpu_devices.split(',')]
    devices = [torch.device(f'cuda:{i}') if torch.cuda.is_available() else torch.device('cpu') for i in gpu_ids]
    logging.info(f"Using devices: {devices}")
    
    # Load model (will be replicated to each GPU in extract_features)
    model = load_trained_model(args.checkpoint_path)
    dataset = AudioDataset(args.data_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    logging.info(f"Found {len(dataset)} audio files")

    # Extract features using multiple GPUs
    features = extract_features(model, dataloader, devices)
    
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
                visualize_features(features, save_path, method='tsne', perplexity=params['perplexity'])
            else:  # umap
                visualize_features(features, save_path, method='umap', 
                                 n_neighbors=params['n_neighbors'], 
                                 min_dist=params['min_dist'])
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