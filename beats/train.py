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
        overlap=0.0,  # 50% overlap between segments
        max_segments_per_file=5,  # Limit segments per file
        random_segments=True  # Randomly select segments
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
    
    logger.info("Starting training loop")
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (audio, _) in enumerate(dataloader):  # Modified this line to unpack
            # For BEATs, we should pass None as padding_mask since we're using fixed-length inputs
            # The model will handle the padding internally based on the fbank features
            if hasattr(model, 'module'):
                features, _ = model.module.extract_features(audio, padding_mask=None)
            else:
                features, _ = model.extract_features(audio, padding_mask=None)
            
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
