import os
import sys
import argparse
import soundfile as sf
from pathlib import Path
import random

def get_audio_info(file_path):
    """Get number of frames in audio file"""
    try:
        info = sf.info(file_path)
        return int(info.frames)
    except RuntimeError as e:
        print(f"Error processing {file_path}: {e}", file=sys.stderr)
        return 0

def create_manifest(data_dir, split_ratio=0.9):
    """Create train.tsv and eval.tsv manifests"""
    data_dir = Path(data_dir)
    wav_files = list(data_dir.rglob("*.wav"))
    
    # Shuffle and split
    random.shuffle(wav_files)
    split_idx = int(len(wav_files) * split_ratio)
    train_files = wav_files[:split_idx]
    eval_files = wav_files[split_idx:]

    # Create manifests
    for split, files in [("train", train_files), ("eval", eval_files)]:
        manifest_path = data_dir.joinpath(f"{split}.tsv")
        with open(manifest_path, "w") as f:
            print("path\tframes", file=f)
            for wav_path in files:
                frames = get_audio_info(wav_path)
                rel_path = wav_path.relative_to(data_dir)
                print(f"{rel_path}\t{frames}", file=f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Directory containing wav files")
    parser.add_argument("--split_ratio", type=float, default=0.9, help="Train/eval split ratio (default: 0.9)")
    args = parser.parse_args()
    
    create_manifest(args.data_dir, args.split_ratio)

if __name__ == "__main__":
    main()