import numpy as np
import torch
from pathlib import Path
import gradio as gr
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
import threading

# Import from original vis.py
from vis import load_trained_model, parse_grid_params, DEFAULT_GRID
from dataset import AudioDataset
from torch.utils.data import DataLoader

# Global variables to store features and status
global_features = None
global_filenames = None
extraction_complete = False
progress_percent = 0

def extract_features(model, dataloader, device, progress_callback):
    global global_features, global_filenames, extraction_complete, progress_percent
    
    features = []
    filenames = []
    total_batches = len(dataloader)
    
    with torch.no_grad():
        for i, (audio, fname) in enumerate(dataloader):
            audio = audio.to(device)
            feat, _ = model.extract_features(audio, padding_mask=None)
            feat = torch.mean(feat, dim=1)
            features.append(feat.cpu().numpy())
            filenames.extend(fname)
            
            # Update progress
            progress_percent = (i + 1) / total_batches * 100
            progress_callback(f"Extracting features... {progress_percent:.1f}%")
    
    global_features = np.concatenate(features, axis=0)
    global_filenames = filenames
    extraction_complete = True
    progress_callback("Feature extraction complete!")

def create_visualization(method, perplexity, n_neighbors, min_dist):
    if not extraction_complete:
        return None
    
    plt.figure(figsize=(10, 10))
    
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        params_str = f'perplexity={perplexity}'
    else:
        reducer = umap.UMAP(random_state=42, n_neighbors=n_neighbors, min_dist=min_dist)
        params_str = f'n_neighbors={n_neighbors}, min_dist={min_dist}'

    embedded = reducer.fit_transform(global_features)
    
    plt.scatter(embedded[:, 0], embedded[:, 1], alpha=0.5)
    plt.title(f'Audio Features Visualization\n{method.upper()}\n({params_str})')
    
    fig = plt.gcf()
    plt.close()
    return fig

def main():
    import argparse
    import json
    
    # Use the same argument parser as in vis.py
    parser = argparse.ArgumentParser(description='Interactive visualization of audio features')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Directory containing the audio files to process')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                      help='Path to the trained BEATs model checkpoint (.pt file)')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Number of audio files to process simultaneously. Larger values use more memory but process faster')
    parser.add_argument('--output_dir', type=str, default='feature_visualizations',
                      help='Directory where the visualization plots will be saved')
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
    parser.add_argument('--grid_search', type=str, default=json.dumps(DEFAULT_GRID),
                      help='JSON string defining parameter grid. Default explores multiple parameters: '
                           'TSNE perplexity=[5,15,30,50], '
                           'UMAP n_neighbors=[5,15,30] and min_dist=[0.1,0.3,0.5]. '
                           'Example custom grid: '
                           '\'{"tsne": {"perplexity": [10, 30, 50]}, '
                           '"umap": {"n_neighbors": [5, 15, 30], "min_dist": [0.1, 0.5]}}\'')
    args = parser.parse_args()

    # Setup model and data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_trained_model(args.checkpoint_path).to(device)
    dataset = AudioDataset(args.data_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create Gradio interface
    def update_plot(method, perplexity, n_neighbors, min_dist, progress=gr.Progress()):
        if not extraction_complete:
            progress(progress_percent / 100)
            return None
        return create_visualization(method, perplexity, n_neighbors, min_dist)

    iface = gr.Interface(
        fn=update_plot,
        inputs=[
            gr.Radio(["tsne", "umap"], label="Visualization Method", value="tsne"),
            gr.Slider(5, 50, value=30, step=5, label="t-SNE Perplexity"),
            gr.Slider(2, 100, value=15, step=1, label="UMAP n_neighbors"),
            gr.Slider(0.0, 1.0, value=0.1, step=0.1, label="UMAP min_dist")
        ],
        outputs=[gr.Plot()],
        title="Audio Features Visualization",
        description="Visualize audio features using different dimensionality reduction methods",
        live=True
    )
    
    # Start feature extraction in a separate thread
    extraction_thread = threading.Thread(
        target=extract_features,
        args=(model, dataloader, device, print)
    )
    extraction_thread.start()
    
    # Launch the interface
    iface.launch(share=True)
    
    # Wait for extraction to complete before exiting
    extraction_thread.join()

if __name__ == "__main__":
    main()