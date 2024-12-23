import streamlit as st
import subprocess
import os
import torch
from pathlib import Path
import yaml
import tempfile
import re
import time
import signal
from dataclasses import dataclass
from typing import Optional, Dict
import psutil  # Add this import
import atexit

@dataclass
class TrainingMetrics:
    epoch: int = 0
    batch: int = 0
    loss: float = 0.0
    epoch_loss: float = 0.0
    progress: float = 0.0
    
class TrainingMonitor:
    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.metrics = TrainingMetrics()
        self.total_epochs = 0
        self.total_batches = 0
        self.error_patterns = [
            "Error:",
            "Exception:",
            "Traceback",
            "RuntimeError:",
            "CUDA out of memory"
        ]
        self.warning_patterns = [
            "Warning:",
            "UserWarning:",
            "torch.distributed"
        ]
        self._cleanup_handler_registered = False
        
    def _find_process_children(self):
        """Find all child processes of the training process"""
        if not self.process:
            return []
        try:
            parent = psutil.Process(self.process.pid)
            children = parent.children(recursive=True)
            return [parent] + children
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return []
        
    def is_error(self, line: str) -> bool:
        """Check if log line indicates an error"""
        return any(pattern in line for pattern in self.error_patterns)
        
    def is_warning(self, line: str) -> bool:
        """Check if log line is just a warning"""
        return any(pattern in line for pattern in self.warning_patterns)
        
    def parse_log_line(self, line: str) -> Dict:
        """Parse metrics from log line"""
        metrics = {}
        # Match epoch info
        epoch_match = re.search(r"Epoch (\d+)", line)
        if epoch_match:
            metrics['epoch'] = int(epoch_match.group(1))
            
        # Match batch info
        batch_match = re.search(r"Batch (\d+)", line)
        if batch_match:
            metrics['batch'] = int(batch_match.group(1))
            
        # Match loss value
        loss_match = re.search(r"Loss: (\d+\.\d+)", line)
        if loss_match:
            metrics['loss'] = float(loss_match.group(1))
            
        return metrics
    
    def start_training(self, cmd, total_epochs):
        """Start training process with both stdout and stderr pipes"""
        self.total_epochs = total_epochs
        
        # Register cleanup handler only once
        if not self._cleanup_handler_registered:
            atexit.register(self.stop_training)
            self._cleanup_handler_registered = True
            
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1,  # Line buffered
                preexec_fn=os.setsid,
                start_new_session=True  # Create new session
            )
        except Exception as e:
            st.error(f"Failed to start training process: {str(e)}")
            self.process = None
            raise
        
    def stop_training(self):
        """Stop training process and all its children gracefully"""
        if self.process:
            try:
                # Try to stop all child processes
                processes = self._find_process_children()
                
                # Send SIGTERM to process group
                pgid = os.getpgid(self.process.pid)
                os.killpg(pgid, signal.SIGTERM)
                
                # Wait briefly for graceful shutdown
                self.process.wait(timeout=5)
                
            except (ProcessLookupError, psutil.NoSuchProcess):
                pass  # Process already terminated
            except Exception as e:
                st.warning(f"Error during process cleanup: {str(e)}")
            finally:
                # Force kill any remaining processes
                for proc in self._find_process_children():
                    try:
                        proc.kill()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                
                self.process = None
                
    def update_metrics(self, log_line: str):
        """Update metrics from log line"""
        parsed = self.parse_log_line(log_line)
        if parsed:
            if 'epoch' in parsed:
                self.metrics.epoch = parsed['epoch']
            if 'batch' in parsed:
                self.metrics.batch = parsed['batch']
            if 'loss' in parsed:
                self.metrics.loss = parsed['loss']
                
            # Calculate progress
            self.metrics.progress = (self.metrics.epoch + 
                                   (self.metrics.batch / self.total_batches if self.total_batches else 0)) / self.total_epochs

def get_available_gpus():
    if torch.cuda.is_available():
        return [str(i) for i in range(torch.cuda.device_count())]
    return []

def save_config(config):
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as f:
        yaml.dump(config, f)
        return f.name

def configure_accelerate(num_gpus):
    """Configure accelerate using subprocess to simulate user input"""
    config = {
        "compute_environment": "LOCAL_MACHINE",
        "distributed_type": "NO" if num_gpus <= 1 else "MULTI_GPU",
        "num_processes": num_gpus,
        "gpu_ids": "all",
        "main_training_function": "main",
        "mixed_precision": "fp16",
        "rdzv_backend": "static" if num_gpus > 1 else "no",
        "same_network": True,
        "machine_rank": 0
    }
    
    # Save temporary accelerate config
    config_path = os.path.expanduser("~/.cache/huggingface/accelerate/default_config.yaml")
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    
    return config_path

def main():
    st.title("BEATs Training Interface")
    
    # Initialize session state for training monitor
    if 'monitor' not in st.session_state:
        st.session_state.monitor = TrainingMonitor()
    
    with st.sidebar:
        st.header("Training Mode")
        training_mode = st.radio(
            "Select training mode",
            ["Self-supervised", "Supervised", "Pre-training"]
        )
        
        st.header("Hardware Configuration")
        available_gpus = get_available_gpus()
        if available_gpus:
            num_gpus = st.slider("Number of GPUs", 
                                min_value=1, 
                                max_value=len(available_gpus),
                                value=1)
            st.info(f"Will configure distributed training for {num_gpus} GPU{'s' if num_gpus > 1 else ''}")
        else:
            st.warning("No GPUs detected")
            num_gpus = 0
    
    # Main configuration area
    st.header("Data Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        data_dir = st.text_input("Data Directory", "data/unlabeled")
        
        if training_mode == "Supervised":
            labeled_dir = st.text_input("Labeled Data Directory", "data/labeled")
            labels_file = st.text_input("Labels CSV File", "data/labeled/labels.csv")
        
        if training_mode in ["Self-supervised", "Supervised"]:
            positive_dir = st.text_input("Positive Examples Directory (optional)", "")
            negative_dir = st.text_input("Negative Examples Directory (optional)", "")
    
    with col2:
        output_dir = st.text_input("Output Directory", "runs/beats_training")
        model_path = st.text_input("Pre-trained Model Path (optional)", "")
        
    st.header("Training Parameters")
    col3, col4, col5 = st.columns(3)
    
    with col3:
        epochs = st.number_input("Number of Epochs", min_value=1, value=10)
        batch_size = st.number_input("Batch Size", min_value=1, value=32)
        learning_rate = st.number_input("Learning Rate", value=1e-5, format="%.1e")
        
    with col4:
        segment_duration = st.number_input("Segment Duration (seconds)", min_value=1, value=10)
        checkpoint_freq = st.number_input("Checkpoint Frequency (epochs)", min_value=1, value=1)
        grad_accum_steps = st.number_input("Gradient Accumulation Steps", min_value=1, value=1)
        
    with col5:
        if training_mode == "Supervised":
            supervised_weight = st.slider("Supervised Loss Weight", 0.0, 1.0, 0.3)
        
        if training_mode == "Pre-training":
            mask_ratio = st.slider("Mask Ratio", 0.0, 1.0, 0.75)
            target_length = st.number_input("Target Length", min_value=128, value=1024)
    
    st.header("Model Architecture")
    col6, col7 = st.columns(2)
    
    with col6:
        encoder_layers = st.number_input("Encoder Layers", min_value=1, value=12)
        encoder_embed_dim = st.number_input("Encoder Embedding Dimension", min_value=64, value=768)
        
    # Add stop button if training is in progress
    if st.session_state.monitor.process:
        if st.button("Stop Training", key="stop_button"):
            with st.spinner("Stopping training..."):
                st.session_state.monitor.stop_training()
                st.error("Training stopped by user")
                time.sleep(1)  # Brief pause to ensure cleanup
                st.rerun()
    
    # Generate command button
    if st.button("Start Training"):
        try:
            with st.spinner("Configuring Accelerate..."):
                config_path = configure_accelerate(num_gpus)
                st.success(f"Accelerate configured for {num_gpus} GPU{'s' if num_gpus > 1 else ''}")
            
            # Create command
            cmd = ["accelerate", "launch", "beats/train.py"]
            cmd.extend([
                f"--data_dir={data_dir}",
                f"--output_dir={output_dir}",
                f"--batch_size={batch_size}",
                f"--epochs={epochs}",
                f"--lr={learning_rate}",
                f"--segment_duration={segment_duration}",
                f"--checkpoint_freq={checkpoint_freq}",
                f"--gradient_accumulation_steps={grad_accum_steps}",
                f"--encoder_layers={encoder_layers}",
                f"--encoder_embed_dim={encoder_embed_dim}"
            ])
            
            # Optional paths
            if model_path:
                cmd.append(f"--model_path={model_path}")
            if training_mode == "Supervised":
                cmd.extend([
                    f"--labeled_dir={labeled_dir}",
                    f"--supervised_weight={supervised_weight}"
                ])
            if positive_dir:
                cmd.append(f"--positive_dir={positive_dir}")
            if negative_dir:
                cmd.append(f"--negative_dir={negative_dir}")
                
            # Pre-training specific
            if training_mode == "Pre-training":
                cmd.extend([
                    f"--mask_ratio={mask_ratio}",
                    f"--target_length={target_length}"
                ])
            
            # Create persistent UI containers with better layout
            status_container = st.empty()
            col1, col2 = st.columns([2, 1])  # 2:1 ratio for logs and metrics
            
            with col1:
                st.markdown("### Training Logs")
                # Create a large scrollable text area for logs
                log_area = st.empty()
                all_logs = []
            
            with col2:
                st.markdown("### Metrics")
                metric_containers = st.columns(3)
                with metric_containers[0]:
                    epoch_metric = st.empty()
                with metric_containers[1]:
                    batch_metric = st.empty()
                with metric_containers[2]:
                    loss_metric = st.empty()
                progress_bar = st.progress(0)
                metrics_display = st.empty()
            
            status_container.info("Starting training...")
            
            # Start training
            st.session_state.monitor.start_training(cmd, epochs)
            
            has_error = False
            
            # Monitor training progress
            while st.session_state.monitor.process:
                outputs = []
                
                # Read from stdout and stderr (non-blocking)
                for pipe in [st.session_state.monitor.process.stdout, 
                           st.session_state.monitor.process.stderr]:
                    while True:
                        line = pipe.readline()
                        if not line:
                            break
                        outputs.append(line)
                
                # Check if process has finished
                if not outputs and st.session_state.monitor.process.poll() is not None:
                    break
                
                # Process outputs
                for line in outputs:
                    # Add to log history
                    all_logs.append(line.strip())
                    # Keep only last 1000 lines to prevent memory issues
                    if len(all_logs) > 1000:
                        all_logs.pop(0)
                    
                    # Update scrollable log display
                    log_area.code('\n'.join(all_logs))
                    
                    if st.session_state.monitor.is_error(line):
                        has_error = True
                        status_container.error(line.strip())
                    else:
                        # Update metrics if line contains them
                        st.session_state.monitor.update_metrics(line)
                        metrics = st.session_state.monitor.metrics
                        
                        # Update metrics display
                        status_container.info(f"Training in progress... Epoch {metrics.epoch + 1}/{epochs}")
                        epoch_metric.metric("Epoch", f"{metrics.epoch + 1}/{epochs}")
                        batch_metric.metric("Batch", metrics.batch)
                        loss_metric.metric("Loss", f"{metrics.loss:.4f}")
                        progress_bar.progress(metrics.progress)
                        metrics_display.json(st.session_state.monitor.metrics.__dict__)
                
                time.sleep(0.1)
            
            # Check final status
            rc = st.session_state.monitor.process.poll()
            if rc == 0 and not has_error:
                status_container.success("Training completed successfully!")
            else:
                status_container.error("Training failed. Check logs for details.")
            
            # Save logs to file
            log_path = Path(output_dir) / "training_logs.txt"
            with open(log_path, "w") as f:
                f.write('\n'.join(all_logs))
            st.info(f"Full logs saved to: {log_path}")
            
        except Exception as e:
            st.error(f"Error during training: {str(e)}")
            st.session_state.monitor.stop_training()

if __name__ == "__main__":
    main()
