import argparse
import os
from pathlib import Path
import logging

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='Directory containing audio files')
    parser.add_argument('--labels_dir', required=True, help='Directory containing label txt files')
    parser.add_argument('--output_file', default='train.lbl', help='Output .lbl file path')
    return parser.parse_args()

def process_label_file(file_path):
    logging.info(f"Processing label file: {file_path}")
    labels = set()
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                start, end, label = line.strip().split('|')
                labels.update([l.strip() for l in label.split(',')])
    return labels

def convert_labels(data_dir, labels_dir, output_file):
    logging.info(f"Converting labels from {labels_dir} with data in {data_dir}, output to {output_file}")
    data_dir = Path(data_dir)
    labels_dir = Path(labels_dir)
    results = []

    # Find all txt files
    txt_files = list(labels_dir.rglob('*.txt'))
    logging.info(f"Found {len(txt_files)} label files.")
    for txt_path in txt_files:
        # Get corresponding audio path
        rel_path = txt_path.relative_to(labels_dir)
        audio_path = data_dir / rel_path.parent / rel_path.stem
        
        # Skip if audio doesn't exist
        if not (audio_path.with_suffix('.wav').exists() or 
                audio_path.with_suffix('.mp3').exists()):
            logging.warning(f"No audio file found for {txt_path}")
            continue

        # Get relative path from data_dir
        rel_audio_path = audio_path.relative_to(data_dir)
        labels = process_label_file(txt_path)
        
        results.append({
            'path': str(rel_audio_path),
            'labels': ','.join(sorted(labels))
        })

    logging.info(f"Writing output file to {output_file}")
    # Write output file
    with open(output_file, 'w') as f:
        for entry in results:
            f.write(f"{entry['path']}\t{entry['labels']}\n")

def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    convert_labels(args.data_dir, args.labels_dir, args.output_file)

if __name__ == "__main__":
    main()