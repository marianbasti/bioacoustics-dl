import streamlit as st
import subprocess
import os
import torch
from pathlib import Path
import yaml
import tempfile
import time
import re
from datetime import datetime
import queue
import threading
import signal

def get_available_gpus():
    if torch.cuda.is_available():
        return [str(i) for i in range(torch.cuda.device_count())]
    return []

def save_config(config):
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as f:
        yaml.dump(config, f)
        return f.name

class TrainingMonitor:
    def __init__(self):
        self.process = None
        self.output_queue = queue.Queue()
        self.should_stop = False
        self.current_epoch = 0
        self.total_epochs = 0
        self.current_batch = 0
        self.total_batches = 0
        self.current_loss = 0.0
        self.debug_info = []  # Add debug info list
        
    def parse_training_line(self, line):
        """Parse training output for progress information"""
        try:
            # Parse epoch information
            epoch_match = re.search(r'Epoch (\d+)', line)
            if epoch_match:
                self.current_epoch = int(epoch_match.group(1))
            
            # Parse batch information
            batch_match = re.search(r'Batch (\d+)', line)
            if batch_match:
                self.current_batch = int(batch_match.group(1))
            
            # Parse loss information
            loss_match = re.search(r'Loss: (\d+\.\d+)', line)
            if loss_match:
                self.current_loss = float(loss_match.group(1))
                
        except Exception:
            pass  # Silently handle parsing errors
        
    def stream_output(self, process):
        """Stream process output to queue"""
        try:
            for line in iter(process.stdout.readline, ''):
                if self.should_stop:
                    break
                self.parse_training_line(line)
                self.output_queue.put(('stdout', line))
                self.debug_info.append(f"STDOUT: {line.strip()}")
            process.stdout.close()
        except Exception as e:
            self.debug_info.append(f"Error in stdout stream: {str(e)}")
            
        try:
            for line in iter(process.stderr.readline, ''):
                if self.should_stop:
                    break
                self.output_queue.put(('stderr', line))
                self.debug_info.append(f"STDERR: {line.strip()}")
            process.stderr.close()
        except Exception as e:
            self.debug_info.append(f"Error in stderr stream: {str(e)}")

    def start_training(self, cmd):
        """Start training process with output monitoring"""
        self.should_stop = False
        self.debug_info.append(f"Starting training with command: {' '.join(cmd)}")
        
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                preexec_fn=os.setsid,
                cwd=os.path.dirname(os.path.abspath(__file__))  # Set working directory to script location
            )
            self.debug_info.append(f"Process started with PID: {self.process.pid}")
            
            # Start output streaming threads
            threading.Thread(target=self.stream_output, args=(self.process,), daemon=True).start()
            
        except Exception as e:
            self.debug_info.append(f"Error starting process: {str(e)}")
            raise

    def stop_training(self):
        """Gracefully stop training process"""
        if self.process:
            self.should_stop = True
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            self.process.wait()
            self.process = None

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
    # Initialize training monitor in session state if not exists
    if 'monitor' not in st.session_state:
        st.session_state.monitor = TrainingMonitor()
        st.session_state.cmd = None
        st.session_state.config_saved = False
    
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
        
    col_buttons = st.columns(2)
    
    # Generate command button in first column
    if col_buttons[0].button("Generate Training Command"):
        cmd = ["accelerate", "launch", "--config_file", "~/.cache/huggingface/accelerate/default_config.yaml", "train.py"]
        
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
    
    if st.session_state.monitor.process and st.session_state.monitor.process.poll() is None:
        # Show stop button if training is running
        if col_buttons[1].button("Stop Training"):
            st.session_state.monitor.stop_training()
            st.warning("Training stopped by user")
    else:
        # Show start button if training is not running
        start_disabled = not st.session_state.config_saved
        if col_buttons[1].button("Start Training", disabled=start_disabled):
            if st.session_state.cmd:
                try:
                    with st.spinner("Configuring Accelerate..."):
                        config_path = configure_accelerate(num_gpus)
                        st.success(f"Accelerate configured for {num_gpus} GPU{'s' if num_gpus > 1 else ''}")
                    
                    # Setup training interface
                    ui = render_training_interface()
                    
                    # Add debug output section
                    debug_output = st.empty()
                    
                    # Start training
                    st.session_state.monitor.start_training(st.session_state.cmd)
                    st.session_state.monitor.total_epochs = epochs
                    
                    # Initialize log buffers
                    training_buffer = []
                    warning_buffer = []
                    error_buffer = []
                    
                    training_started = False
                    timeout_counter = 0
                    
                    # Monitor training progress
                    while (st.session_state.monitor.process and 
                           st.session_state.monitor.process.poll() is None):
                        try:
                            # Show debug info
                            debug_output.code('\n'.join(st.session_state.monitor.debug_info))
                            
                            # Check if process has started producing output
                            if not training_started:
                                if timeout_counter > 50:  # 5 seconds timeout
                                    st.error("Training process failed to start producing output")
                                    st.code('\n'.join(st.session_state.monitor.debug_info))
                                    break
                                timeout_counter += 1
                            
                            # Get output from queue
                            try:
                                stream, line = st.session_state.monitor.output_queue.get_nowait()
                                training_started = True  # We got output
                                
                                # Process output
                                if stream == 'stdout':
                                    training_buffer.append(line)
                                    if len(training_buffer) > 100:
                                        training_buffer.pop(0)
                                    ui['training_log'].code('\n'.join(training_buffer))
                                else:  # stderr
                                    if 'warning' in line.lower():
                                        warning_buffer.append(line)
                                        ui['warning_log'].code('\n'.join(warning_buffer))
                                    else:
                                        error_buffer.append(line)
                                        ui['error_log'].code('\n'.join(error_buffer))
                                
                            except queue.Empty:
                                time.sleep(0.1)
                                continue
                                
                        except Exception as e:
                            st.error(f"Error monitoring training: {str(e)}")
                            st.code('\n'.join(st.session_state.monitor.debug_info))
                            break
                    
                    # Check final status
                    return_code = st.session_state.monitor.process.returncode if st.session_state.monitor.process else None
                    st.write(f"Process return code: {return_code}")
                    
                    if return_code == 0:
                        st.success("Training completed successfully!")
                    else:
                        st.error("Training failed. Check debug output below:")
                        st.code('\n'.join(st.session_state.monitor.debug_info))
                        
                except Exception as e:
                    st.error(f"Error during training setup: {str(e)}")
                    if st.session_state.monitor.debug_info:
                        st.code('\n'.join(st.session_state.monitor.debug_info))
            else:
                st.error("Please generate training command first")

if __name__ == "__main__":
    main()
