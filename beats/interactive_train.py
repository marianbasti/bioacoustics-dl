import streamlit as st
import subprocess
import os
import torch
from pathlib import Path
import yaml
import tempfile

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

def monitor_training(process, log_placeholder, progress_bar):
    """Monitor training progress and update Streamlit UI"""
    try:
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            
            if output:
                # Parse progress information
                if "Epoch" in output and "Batch" in output and "Loss:" in output:
                    try:
                        # Extract epoch and loss info
                        parts = output.split()
                        epoch = int(parts[parts.index("Epoch") + 1])
                        loss = float(parts[parts.index("Loss:") + 1])
                        
                        # Update progress bar
                        progress = (epoch) / total_epochs
                        progress_bar.progress(progress)
                        
                        # Show current stats
                        stats_cols = st.columns(3)
                        with stats_cols[0]:
                            st.metric("Current Epoch", epoch)
                        with stats_cols[1]:
                            st.metric("Current Loss", f"{loss:.4f}")
                        with stats_cols[2]:
                            st.metric("Progress", f"{progress*100:.1f}%")
                            
                    except (ValueError, IndexError):
                        pass
                
                # Always update log
                log_placeholder.text(output.strip())
    except Exception as e:
        st.error(f"Error monitoring training: {str(e)}")

def main():
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
        
    # Generate command button
    if st.button("Generate Training Command"):
        cmd = ["accelerate", "launch", "train.py"]
        
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
        
        # Display command
        st.code(" ".join(cmd))
        
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
        st.success(f"Configuration saved to: {config_path}")
        
        # Execute button
        if st.button("Start Training"):
            try:
                with st.spinner("Configuring Accelerate..."):
                    config_path = configure_accelerate(num_gpus)
                    st.success(f"Accelerate configured for {num_gpus} GPU{'s' if num_gpus > 1 else ''}")
                
                st.info("Starting training...")
                
                # Create container for training monitoring
                monitor_container = st.container()
                with monitor_container:
                    st.subheader("Training Progress")
                    progress_bar = st.progress(0.0)
                    log_placeholder = st.empty()
                    
                    # Create expander for detailed logs
                    with st.expander("Detailed Logs", expanded=False):
                        detailed_logs = st.empty()
                
                # Launch training process
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    bufsize=1  # Line buffered
                )
                
                # Monitor training in the UI
                monitor_training(process, log_placeholder, progress_bar)
                
                # Check final status
                rc = process.poll()
                if rc == 0:
                    st.success("Training completed successfully!")
                    
                    # Show final model location
                    final_model = os.path.join(output_dir, "checkpoint_final.pt")
                    if os.path.exists(final_model):
                        st.info(f"Final model saved to: {final_model}")
                else:
                    st.error("Training failed. Check detailed logs for more information.")
                    
                    # Show error output if available
                    error_output = process.stderr.read()
                    if error_output:
                        with st.expander("Error Details"):
                            st.code(error_output)
                    
            except Exception as e:
                st.error(f"Error during training: {str(e)}")
                raise e
            finally:
                # Cleanup process if still running
                if 'process' in locals() and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()

if __name__ == "__main__":
    main()
