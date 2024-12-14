
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from mplcursors import cursor
import numpy as np
import torch
from pathlib import Path

# Import from original vis.py
from vis import load_trained_model, parse_grid_params, DEFAULT_GRID
from dataset import AudioDataset
from torch.utils.data import DataLoader

class RealTimeVisualizerWindow(QMainWindow):
    def __init__(self, features, filenames, method='tsne', perplexity=30, 
                 n_neighbors=15, min_dist=0.1):
        super().__init__()
        self.setWindowTitle("Audio Features Visualization")
        self.setGeometry(100, 100, 800, 800)

        # Create the main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Create the figure and canvas
        self.figure = Figure(figsize=(8, 8))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Initialize the plot
        self.ax = self.figure.add_subplot(111)
        self.scatter = None
        self.features = features
        self.filenames = filenames
        
        # Start visualization
        self.visualize(method, perplexity, n_neighbors, min_dist)

    def visualize(self, method='tsne', perplexity=30, n_neighbors=15, min_dist=0.1):
        from sklearn.manifold import TSNE
        import umap
        
        # Initialize reducer based on method
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            params_str = f'perplexity={perplexity}'
        else:
            reducer = umap.UMAP(random_state=42, n_neighbors=n_neighbors, min_dist=min_dist)
            params_str = f'n_neighbors={n_neighbors}, min_dist={min_dist}'

        # Fit and transform the data
        embedded = reducer.fit_transform(self.features)
        
        # Plot the points
        self.ax.clear()
        self.scatter = self.ax.scatter(embedded[:, 0], embedded[:, 1], alpha=0.5)
        self.ax.set_title(f'Audio Features Visualization\n{method.upper()}\n({params_str})')
        
        # Add interactive hover annotations
        cursor = mplcursors.cursor(self.scatter, hover=True)
        @cursor.connect("add")
        def on_hover(sel):
            point_idx = sel.target.index
            sel.annotation.set_text(Path(self.filenames[point_idx]).name)
        
        self.canvas.draw()

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

    # Initialize PyQt application
    app = QApplication(sys.argv)
    
    # Setup model and data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_trained_model(args.checkpoint_path).to(device)
    dataset = AudioDataset(args.data_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # Extract features and get filenames
    features = []
    filenames = []
    
    with torch.no_grad():
        for audio, fname in dataloader:
            audio = audio.to(device)
            feat, _ = model.extract_features(audio, padding_mask=None)
            feat = torch.mean(feat, dim=1)
            features.append(feat.cpu().numpy())
            filenames.extend(fname)
    
    features = np.concatenate(features, axis=0)
    
    # Create and show the visualization window
    window = RealTimeVisualizerWindow(features, filenames,
                                    method=args.methods[0],
                                    perplexity=args.perplexity,
                                    n_neighbors=args.n_neighbors,
                                    min_dist=args.min_dist)
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()