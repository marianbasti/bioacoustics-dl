import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
from pathlib import Path
from torch.utils.data import DataLoader
from BEATs import BEATs, BEATsConfig
from dataset import AudioDataset
import argparse

def load_trained_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    cfg = BEATsConfig(checkpoint['cfg'])
    model = BEATs(cfg)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model

def extract_features(model, dataloader, device='cuda'):
    features_list = []
    
    with torch.no_grad():
        for audio in dataloader:
            audio = audio.to(device)
            features, _ = model.extract_features(audio, padding_mask=None)
            # Average pool temporal dimension
            features = torch.mean(features, dim=1)
            features_list.append(features.cpu().numpy())
    
    return np.concatenate(features_list, axis=0)

def visualize_features(features, save_path, method='tsne', perplexity=30, n_neighbors=15, min_dist=0.1):
    # Dimensionality reduction
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    else:
        reducer = umap.UMAP(random_state=42, n_neighbors=n_neighbors, min_dist=min_dist)
    
    embedded = reducer.fit_transform(features)
    
    # Create visualization
    plt.figure(figsize=(10, 10))
    plt.scatter(embedded[:, 0], embedded[:, 1], alpha=0.5)
    plt.title(f'Audio Features Visualization ({method.upper()})')
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize audio features using BEATs model')
    # Data and model parameters
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing audio files')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for feature extraction')
    parser.add_argument('--output_dir', type=str, default='feature_visualizations', help='Output directory for visualizations')
    
    # Visualization parameters
    parser.add_argument('--methods', nargs='+', choices=['tsne', 'umap'], default=['tsne', 'umap'], 
                      help='Visualization methods to use')
    parser.add_argument('--perplexity', type=float, default=30, help='Perplexity parameter for t-SNE')
    parser.add_argument('--n_neighbors', type=int, default=15, help='n_neighbors parameter for UMAP')
    parser.add_argument('--min_dist', type=float, default=0.1, help='min_dist parameter for UMAP')
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model and data
    model = load_trained_model(args.checkpoint_path).to(device)
    dataset = AudioDataset(args.data_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # Extract features
    features = extract_features(model, dataloader, device)
    
    # Create visualizations
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
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