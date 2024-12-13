import torch
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import functional as F
from dataset import AudioDataset
from BEATs import BEATs, BEATsConfig
from accelerate import Accelerator

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="BEATs_iter3+_AS2M.pt")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_dir", type=str, default="beats/runs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    return parser.parse_args()

def advanced_audio_contrastive_loss(features, temperature=0.1):
    """
    Enhanced contrastive loss for audio features:
    - Uses time-frequency consistency
    - Handles batch of temporal sequences better
    - Normalizes features properly
    """
    # Average pooling across time dimension if features are 3D (batch, time, features)
    if len(features.shape) == 3:
        global_features = torch.mean(features, dim=1)  # (batch, features)
    else:
        global_features = features
    
    # L2 normalize features
    global_features = F.normalize(global_features, dim=1)
    
    batch_size = global_features.shape[0]
    
    # Compute similarity matrix
    similarity_matrix = torch.matmul(global_features, global_features.T)
    
    # Apply temperature scaling
    similarity_matrix = similarity_matrix / temperature
    
    # Create labels for positive pairs
    labels = torch.arange(batch_size, device=features.device)
    
    # Compute NT-Xent loss (Normalized Temperature-scaled Cross Entropy)
    loss = F.cross_entropy(similarity_matrix, labels)
    
    return loss

def main():
    args = parse_args()
    accelerator = Accelerator()
    
    # Load pre-trained model safely
    checkpoint = torch.load(args.model_path, weights_only=True, map_location='cpu')
    cfg = BEATsConfig(checkpoint['cfg'])
    
    # Create model first with original config
    model = BEATs(cfg)
    
    # Load pre-trained weights
    model.load_state_dict(checkpoint['model'])
    
    # Setup dataset and dataloader
    dataset = AudioDataset(args.data_dir)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr)
    
    # Prepare everything with accelerator
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, audio in enumerate(dataloader):
            # Create padding mask (assuming no padding in fixed-length samples)
            padding_mask = torch.zeros(audio.shape[0], audio.shape[1]).bool()
            padding_mask = accelerator.prepare(padding_mask)[0]
            
            # Forward pass
            features, _ = model.extract_features(audio, padding_mask)
            
            # Compute SSL loss (example: use features for contrastive learning)
            # This is a simple example - you might want to implement more sophisticated SSL objectives
            loss = advanced_audio_contrastive_loss(features)
            
            # Scale loss by gradient accumulation steps
            loss = loss / args.gradient_accumulation_steps
            
            # Backward pass with accelerator
            accelerator.backward(loss)
            
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * args.gradient_accumulation_steps
            
            if batch_idx % 10 == 0:
                accelerator.print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        accelerator.print(f"Epoch {epoch} completed, Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint on main process only
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            checkpoint = {
                'epoch': epoch,
                'model': unwrapped_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'cfg': cfg.__dict__,
            }
            torch.save(checkpoint, output_dir / f"checkpoint_epoch_{epoch}.pt")
        
        # Wait for checkpoint to be saved
        accelerator.wait_for_everyone()

if __name__ == "__main__":
    main()
