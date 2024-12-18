import logging
import torch
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import functional as F
from dataset import AudioDataset
from BEATs import BEATs, BEATsConfig
from accelerate import Accelerator

class MemoryBank:
    def __init__(self, size=4096, feature_dim=768, device='cuda'):
        """
        Args:
            size: Number of features to store
            feature_dim: Dimension of each feature vector
            device: Device to store the features on
        """
        self.size = size
        self.feature_dim = feature_dim
        self.bank = torch.randn(size, feature_dim, device=device)
        self.bank = F.normalize(self.bank, dim=1)
        self.pointer = 0

    def update(self, features):
        """Update memory bank with new features"""
        batch_size = features.shape[0]
        features = F.normalize(features.detach(), dim=1)
        
        # Update memory bank
        if self.pointer + batch_size >= self.size:
            self.pointer = 0
        
        self.bank[self.pointer:self.pointer + batch_size] = features
        self.pointer = (self.pointer + batch_size) % self.size

    def get_memory(self):
        """Return current memory bank"""
        return self.bank
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, default=None,
                      help="Path to pre-trained model (optional)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_dir", type=str, default="beats/runs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--checkpoint_freq", type=int, default=1,
                      help="Save checkpoint every N epochs")
    parser.add_argument("--encoder_layers", type=int, default=12,
                      help="Number of encoder layers when training from scratch")
    parser.add_argument("--encoder_embed_dim", type=int, default=768,
                      help="Encoder embedding dimension when training from scratch")
    return parser.parse_args()

def advanced_audio_contrastive_loss(features, temperature=0.1, memory_bank=None, mask=None):
    """
    Enhanced contrastive loss for audio features:
    - Uses time-frequency consistency
    - Handles batch of temporal sequences better
    - Normalizes features properly
    - Memory bank support
    - Hard negative mining
    - Local-global consistency
    - Symmetric loss
    - Masked padding support
    """
    # Handle 3D features (batch, time, features)
    if len(features.shape) == 3:
        # Global features via average pooling
        global_features = torch.mean(features, dim=1)  # (batch, features)
        # Local features for local-global consistency
        local_features = features
    else:
        global_features = features
        local_features = None
    
    # L2 normalize features
    global_features = F.normalize(global_features, dim=1)
    batch_size = global_features.shape[0]
    
    # Compute similarity matrix
    sim_matrix = torch.matmul(global_features, global_features.T)
    sim_matrix = sim_matrix / temperature
    
    # Apply mask if provided
    if mask is not None:
        sim_matrix = sim_matrix * mask
    
    # Add memory bank negatives if provided
    if memory_bank is not None:
        memory_bank = F.normalize(memory_bank, dim=1)
        neg_sim = torch.matmul(global_features, memory_bank.T) / temperature
        sim_matrix = torch.cat([sim_matrix, neg_sim], dim=1)
    
    # Labels for positive pairs
    labels = torch.arange(batch_size, device=features.device)
    
    # Symmetric loss calculation
    loss = F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.T, labels)
    loss = loss / 2
    
    # Add local-global consistency if we have local features
    if local_features is not None:
        local_features = F.normalize(local_features, dim=2)
        local_global_sim = torch.matmul(
            local_features, global_features.unsqueeze(2)
        ).squeeze(2)
        local_global_loss = F.cross_entropy(local_global_sim / temperature, 
                                          torch.arange(batch_size, device=features.device))
        loss = loss + 0.5 * local_global_loss
    
    return loss

def main():
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(Path(args.output_dir) / "training.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting training with arguments: {vars(args)}")
    accelerator = Accelerator()
    
    if args.model_path:
        logger.info(f"Loading pre-trained model from {args.model_path}")
        # Load pre-trained model
        checkpoint = torch.load(args.model_path, weights_only=True, map_location='cpu')
        cfg = BEATsConfig(checkpoint['cfg'])
        model = BEATs(cfg)
        model.load_state_dict(checkpoint['model'])
    else:
        logger.info("Initializing model from scratch")
        # Initialize from scratch
        cfg = BEATsConfig({
            'encoder_layers': args.encoder_layers,
            'encoder_embed_dim': args.encoder_embed_dim,
            'encoder_ffn_embed_dim': args.encoder_embed_dim * 4,
            'encoder_attention_heads': args.encoder_embed_dim // 64,
            'input_patch_size': 16,  # common default value
            'embed_dim': 512,  # common default value
        })
        model = BEATs(cfg)
    
    # Setup dataset and dataloader
    logger.info(f"Loading dataset from {args.data_dir}")
    dataset = AudioDataset(
        root_dir=args.data_dir,
        segment_duration=10,
        overlap=0.01,  # 1% overlap between segments
        max_segments_per_file=6,  # Limit segments per file
        random_segments=False  # Randomly select segments
    )
    logger.info(f"Dataset size: {len(dataset)} files")
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
    
    # Initialize memory bank
    memory_bank = MemoryBank(
        size=4096,  # Store 4096 negative examples
        feature_dim=args.encoder_embed_dim,  # Match model dimension
        device=accelerator.device
    )

    logger.info("Starting training loop")
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (audio, _) in enumerate(dataloader):
            if hasattr(model, 'module'):
                features, _ = model.module.extract_features(audio, padding_mask=None)
            else:
                features, _ = model.extract_features(audio, padding_mask=None)
            
            # Get global features
            global_features = torch.mean(features, dim=1)

            # Compute SSL loss (example: use features for contrastive learning)
            loss = advanced_audio_contrastive_loss(features,memory_bank=memory_bank.get_memory())
            
            # Scale loss by gradient accumulation steps
            loss = loss / args.gradient_accumulation_steps
            
            # Update memory bank with current batch
            memory_bank.update(global_features)

            # Backward pass with accelerator
            accelerator.backward(loss)
            
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * args.gradient_accumulation_steps
            
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch} completed, Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint on main process only
        if accelerator.is_main_process and (epoch + 1) % args.checkpoint_freq == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
            logger.info(f"Saving checkpoint to {checkpoint_path}")
            unwrapped_model = accelerator.unwrap_model(model)
            checkpoint = {
                'epoch': epoch,
                'model': unwrapped_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'cfg': cfg.__dict__,
            }
            torch.save(checkpoint, checkpoint_path)
        
        # Wait for checkpoint to be saved
        accelerator.wait_for_everyone()

if __name__ == "__main__":
    main()
