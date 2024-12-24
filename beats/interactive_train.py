import streamlit as st
import os
import torch
import yaml
import tempfile
import time
from datetime import datetime
import queue
import threading
from train import main as train_main  # Import the training function
from dataclasses import dataclass
from typing import Optional, Callable

@dataclass
class TrainingCallback:
    """Callback interface for training progress updates"""
    on_epoch_start: Optional[Callable[[int], None]] = None
    on_epoch_end: Optional[Callable[[int, float], None]] = None
    on_batch_end: Optional[Callable[[int, int, float], None]] = None
    on_training_end: Optional[Callable[[float], None]] = None
    should_stop: bool = False

class TrainingMonitor:
    def __init__(self):
        self.current_epoch = 0
        self.total_epochs = 0
        self.current_batch = 0
        self.total_batches = 0
        self.current_loss = 0.0
        self.training_thread = None
        self.callback = None
        self.output_queue = queue.Queue()
        self.should_stop = False

    def create_callback(self):
        """Create training callback"""
        def on_epoch_start(epoch):
            self.current_epoch = epoch
            self.output_queue.put(('info', f'Starting epoch {epoch}'))

        def on_epoch_end(epoch, loss):
            self.current_loss = loss
            self.output_queue.put(('info', f'Epoch {epoch} finished with loss: {loss:.4f}'))

        def on_batch_end(epoch, batch, loss):
            self.current_batch = batch
            self.current_loss = loss
            self.output_queue.put(('info', f'Epoch {epoch}, Batch {batch}, Loss: {loss:.4f}'))

        self.callback = TrainingCallback(
            on_epoch_start=on_epoch_start,
            on_epoch_end=on_epoch_end,
            on_batch_end=on_batch_end
        )
        return self.callback

    def start_training(self, config):
        """Start training in a separate thread"""
        def training_thread():
            try:
                train_main(config, self.create_callback())
            except Exception as e:
                self.output_queue.put(('error', f'Training failed: {str(e)}'))
            finally:
                self.output_queue.put(('info', 'Training finished'))

        self.training_thread = threading.Thread(target=training_thread)
        self.training_thread.start()

    def stop_training(self):
        """Stop training gracefully"""
        if self.callback:
            self.callback.should_stop = True
        if self.training_thread:
            self.training_thread.join()
        self.training_thread = None
        self.callback = None

def get_available_gpus():
    if torch.cuda.is_available():
        return [str(i) for i in range(torch.cuda.device_count())]
    return []

def save_config(config):
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as f:
        yaml.dump(config, f)
        return f.name

def configure_accelerate(num_gpus, output_dir):
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
    
    # Save accelerate config to the output directory
    config_path = os.path.expanduser(f"{output_dir}/accelerate_config.yaml")
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    
    return config_path

def render_training_interface():
    """Render training progress interface"""
    # Create columns for metrics
    col1, col2, col3 = st.columns(3)
    
    # Progress metrics
    progress = col1.empty()
    epoch_status = col2.empty()
    loss_metric = col3.empty()
    
    # Training log sections
    log_tabs = st.tabs(["Training Log", "Warnings", "Errors"])
    training_log = log_tabs[0].empty()
    warning_log = log_tabs[1].empty()
    error_log = log_tabs[2].empty()
    
    return {
        'progress': progress,
        'epoch_status': epoch_status,
        'loss_metric': loss_metric,
        'training_log': training_log,
        'warning_log': warning_log,
        'error_log': error_log
    }

def main():
    # Initialize training monitor and button state in session state if not exists
    if 'monitor' not in st.session_state:
        st.session_state.monitor = TrainingMonitor()
        st.session_state.cmd = None
        st.session_state.config_saved = False
        st.session_state.training_active = False  # Add training state tracking

    st.title("BEATs Training Interface")
    
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
                                value=len(available_gpus))
            st.info(f"Will configure distributed training for {num_gpus} GPU{'s' if num_gpus > 1 else ''}")
        else:
            st.warning("No GPUs detected")
            num_gpus = 0
    
    # Main configuration area
    st.header("Data Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        data_dir = st.text_input("Data Directory", "")
        
        if training_mode == "Supervised":
            labeled_dir = st.text_input("Labeled Data Directory", "data/labeled")
            labels_file = st.text_input("Labels CSV File", "data/labeled/labels.csv")
        
        if training_mode in ["Self-supervised", "Supervised"]:
            positive_dir = st.text_input("Positive Examples Directory (optional)", "")
            negative_dir = st.text_input("Negative Examples Directory (optional)", "")
    
    with col2:
        output_dir = st.text_input("Output Directory", "")
        model_path = st.text_input("Pre-trained Model Path (optional)", "")
        
    st.header("Training Parameters")
    col3, col4, col5 = st.columns(3)
    
    with col3:
        epochs = st.number_input("Number of Epochs", min_value=1, value=10)
        batch_size = st.number_input("Batch Size", min_value=1, value=2)
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
        
    col_buttons = st.columns(2)
    
    # Generate command button in first column
    if col_buttons[0].button("Generate Training Command"):
        cmd = ["accelerate", "launch", "--config_file", f"{output_dir}accelerate_config.yaml", "train.py"]
        
        # Basic parameters
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
        
        # Save configuration
        config = {
            "training_mode": training_mode,
            "data_config": {
                "data_dir": data_dir,
                "output_dir": output_dir,
                "model_path": model_path,
            },
            "training_params": {
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "segment_duration": segment_duration,
                "checkpoint_freq": checkpoint_freq,
                "gradient_accumulation_steps": grad_accum_steps,
            },
            "model_architecture": {
                "encoder_layers": encoder_layers,
                "encoder_embed_dim": encoder_embed_dim,
            }
        }
        
        config_path = save_config(config)
        st.session_state.cmd = cmd
        st.session_state.config_saved = True
        st.code(" ".join(cmd))
        st.success(f"Configuration saved to: {config_path}")
        st.session_state.project_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Combined Start/Stop button with dynamic label
    button_label = "Stop Training" if st.session_state.training_active else "Start Training"
    if col_buttons[1].button(button_label, disabled=not st.session_state.config_saved and not st.session_state.training_active):
        if st.session_state.training_active:
            st.session_state.monitor.stop_training()
            st.session_state.training_active = False
            st.warning("Training stopped by user")
        else:
            try:
                # Create base training config with common parameters
                config = {
                    "training_mode": training_mode,
                    "data_dir": data_dir,
                    "output_dir": output_dir,
                    "batch_size": batch_size,
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                    "segment_duration": segment_duration,
                    "checkpoint_freq": checkpoint_freq,
                    "gradient_accumulation_steps": grad_accum_steps,
                    "encoder_layers": encoder_layers,
                    "encoder_embed_dim": encoder_embed_dim,
                    "num_gpus": num_gpus
                }

                # Add model path only if provided
                if model_path:
                    config["model_path"] = model_path

                # Add mode-specific parameters
                if training_mode == "Supervised":
                    config.update({
                        "labeled_dir": labeled_dir,
                        "supervised_weight": supervised_weight
                    })
                elif training_mode == "Pre-training":
                    config.update({
                        "mask_ratio": mask_ratio,
                        "target_length": target_length
                    })

                # Add optional directories if provided
                if positive_dir:
                    config["positive_dir"] = positive_dir
                if negative_dir:
                    config["negative_dir"] = negative_dir

                # Setup UI components
                ui = render_training_interface()
                
                # Start training
                st.session_state.monitor.start_training(config)
                st.session_state.training_active = True

                # Monitor output queue
                while True:
                    try:
                        msg_type, message = st.session_state.monitor.output_queue.get(timeout=0.1)
                        if msg_type == 'info':
                            ui['training_log'].write(message)
                        elif msg_type == 'error':
                            ui['error_log'].write(message)
                            break
                    except queue.Empty:
                        if not st.session_state.monitor.training_thread.is_alive():
                            break
                        continue

            except Exception as e:
                st.error(f"Error during training: {str(e)}")

if __name__ == "__main__":
    main()
