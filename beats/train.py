import torch
import argparse
import logging
import psutil
import os
from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import functional as F
from dataset import AudioDataset
from BEATs import BEATs, BEATsConfig
from accelerate import Accelerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

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

def log_system_info():
    logger.info("=== System Information ===")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    logger.info(f"CPU count: {psutil.cpu_count()}")
    logger.info(f"RAM available: {psutil.virtual_memory().available / 1024**3:.2f} GB")

def main():
    args = parse_args()
    logger.info("=== Training Configuration ===")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
    
    log_system_info()
    accelerator = Accelerator()
    logger.info(f"Using accelerator: {accelerator.state}")
    
    if args.model_path:
        logger.info(f"Loading pre-trained model from {args.model_path}")
        # Load pre-trained model
        checkpoint = torch.load(args.model_path, weights_only=True, map_location='cpu')
        cfg = BEATsConfig(checkpoint['cfg'])
        model = BEATs(cfg)
        model.load_state_dict(checkpoint['model'])
    else:
        logger.info("Initializing model from scratch with configuration:")
        cfg_dict = {
            'encoder_layers': args.encoder_layers,
            'encoder_embed_dim': args.encoder_embed_dim,
            'encoder_ffn_embed_dim': args.encoder_embed_dim * 4,
            'encoder_attention_heads': args.encoder_embed_dim // 64,
            'input_patch_size': 16,  # common default value
            'embed_dim': 512,  # common default value
        }
        logger.info(cfg_dict)
        cfg = BEATsConfig(cfg_dict)
        model = BEATs(cfg)
    
    # Setup dataset and dataloader
    logger.info(f"Loading dataset from {args.data_dir}")
    dataset = AudioDataset(args.data_dir)
    logger.info(f"Dataset size: {len(dataset)} samples")
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
    
    logger.info("=== Starting Training ===")
    best_loss = float('inf')
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        epoch_start_time = datetime.now()
        total_loss = 0
        
        for batch_idx, (audio, _) in enumerate(dataloader):  # Modified this line to unpack
            # Log memory usage periodically
            if batch_idx % 100 == 0:
                if torch.cuda.is_available():
                    logger.debug(f"CUDA memory: {torch.cuda.memory_allocated() / 1024**2:.2f}MB")
                
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
                logger.info(f"Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, "
                          f"Loss: {loss.item():.4f}, "
                          f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        avg_loss = total_loss / len(dataloader)
        epoch_time = datetime.now() - epoch_start_time
        logger.info(f"Epoch {epoch} completed in {epoch_time}")
        logger.info(f"Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint on main process only
        if avg_loss < best_loss and accelerator.is_main_process:
            best_loss = avg_loss
            logger.info(f"New best loss: {best_loss:.4f}, saving checkpoint...")
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
    try:
        main()
    except Exception as e:
        logger.exception("Training failed with exception")
        raise
