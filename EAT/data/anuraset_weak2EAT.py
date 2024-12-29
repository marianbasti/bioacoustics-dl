import csv
import argparse
import os
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def extract_labels_from_file(input_file, audio_dir):
    """
    Extract labels dynamically from the input CSV file. The function will:
    - Read the species columns from the header (excluding the first two columns).
    - Process each row to extract the filename and present species.
    
    Args:
    - input_file (str): The path to the CSV file containing the data.
    - audio_dir (str): The base directory for audio files.
    
    Returns:
    - list of tuples: Each tuple contains a filename and a comma-separated string of species labels.
    """
    logging.info(f"Reading labels from {input_file}")
    extracted_data = []
    
    with open(input_file, mode='r', newline='') as file:
        reader = csv.reader(file)
        headers = next(reader)  # Read the header
        
        # Extract species column names (skip the first two columns: MONITORING_SITE, AUDIO_FILE_ID)
        species_columns = headers[2:]
        logging.info(f"Found {len(species_columns)} species columns")
        
        row_count = 0
        for row in reader:
            # Create relative path instead of full path
            filename = os.path.normpath(f"{row[0]}/{row[1]}.wav")
            labels = []
            
            # Iterate over the species columns (starting from index 2)
            for i, value in enumerate(row[2:], start=0):  # Start at index 0 for species columns
                # Store 0..3: if blank, treat as '0' = absent
                level = value if value in ['1','2','3'] else '0'
                labels.append(f"{species_columns[i]}={level}")
            
            # Join the labels into a single string, separated by commas
            labels_str = ','.join(labels)
            extracted_data.append((filename, labels_str))
            row_count += 1
        
        logging.info(f"Processed {row_count} entries from CSV file")
    return extracted_data

def get_unique_labels(extracted_data):
    logging.info("Extracting unique labels")
    # Get all unique labels
    unique_labels = set()
    for _, labels_str in extracted_data:
        labels = labels_str.split(',')
        unique_labels.update(labels)
    logging.info(f"Found {len(unique_labels)} unique labels")
    return sorted(list(unique_labels))

def write_label_descriptors(labels, output_path):
    """Write unique species names (without levels) to label descriptors file"""
    unique_species = sorted(list({label.split('=')[0] for label in labels}))
    with open(output_path, "w") as f:
        for idx, species in enumerate(unique_species):
            f.write(f"{idx},{species}\n")

def collect_audio_files(audio_dir):
    logging.info(f"Collecting audio files from {audio_dir}")
    import soundfile as sf
    audio_files = []
    
    skipped_files = 0
    for root, _, files in os.walk(audio_dir):
        for file in files:
            if file.endswith(('.wav', '.WAV')):
                full_path = os.path.join(root, file)
                try:
                    info = sf.info(full_path)
                    num_samples = int(info.frames)
                except Exception as e:
                    skipped_files += 1
                    logging.error(f"Error reading {full_path}: {str(e)}")
                    continue
                rel_path = os.path.relpath(full_path, audio_dir)
                audio_files.append((rel_path, num_samples))
    
    logging.info(f"Found {len(audio_files)} valid audio files. Skipped {skipped_files} files with errors.")
    return audio_files

def write_label_files(extracted_data, output_dir, audio_dir):
    logging.info("Writing label files")
    
    # First get valid audio files
    audio_files = {os.path.normpath(path): samples 
                  for path, samples in collect_audio_files(audio_dir)}
    
    def write_set(data, output_file):
        with open(os.path.join(output_dir, output_file), "w") as f:
            for filename, labels in data:
                # Check if audio file exists
                norm_path = os.path.normpath(filename)
                if norm_path not in audio_files:
                    continue
                    
                base_filename = os.path.splitext(os.path.basename(filename))[0]
                
                # Convert to species-level format
                level_dict = {}
                for label in labels.split(','):
                    species, level = label.split('=')
                    level_dict[species] = level
                
                # Write as species=level pairs
                level_labels = [f"{species}={level_dict.get(species, '0')}" 
                              for species in sorted(set(s.split('=')[0] for s in labels.split(',')))]
                if any(l.endswith(('1','2','3')) for l in level_labels):
                    f.write(f"{base_filename}\t{' '.join(level_labels)}\n")

    # Split data
    cutoff = int(0.8 * len(extracted_data))
    train_data = extracted_data[:cutoff]
    eval_data = extracted_data[cutoff:]
    
    write_set(train_data, "train.lbl")
    write_set(eval_data, "eval.lbl")
    logging.info("Finished writing label files")

def write_tsv_files(extracted_data, audio_dir, output_dir):
    logging.info("Writing TSV files")
    """Write train.tsv and eval.tsv with recursive audio paths."""
    audio_files = {}
    for path, samples in collect_audio_files(audio_dir):
        normalized_path = os.path.normpath(path)
        audio_files[normalized_path] = (path, samples)
    
    logging.info(f"Found {len(audio_files)} audio files for TSV generation")
    def write_set(data, output_file):
        with open(os.path.join(output_dir, output_file), "w") as f:
            f.write(f"{audio_dir}\n")
            for filename, _ in data:
                rel_path = os.path.normpath(filename)
                if rel_path in audio_files:
                    path, num_samples = audio_files[rel_path]
                    base_filename = os.path.basename(path)
                    f.write(f"{base_filename}\t{num_samples}\n")
    
    # Split data
    cutoff = int(0.8 * len(extracted_data))
    train_data = extracted_data[:cutoff]
    eval_data = extracted_data[cutoff:]
    
    # Write both TSV and LBL files
    write_set(train_data, "train.tsv")
    write_set(eval_data, "eval.tsv")
    logging.info("Finished writing TSV files")

def main():
    parser = argparse.ArgumentParser(description="Extract labels from a CSV file.")
    parser.add_argument("--input_file", required=True, help="Path to the input CSV file.")
    parser.add_argument("--output_dir", help="Directory to store output files.")
    parser.add_argument("--audio_dir", help="Directory containing the audio files.")
    args = parser.parse_args()
    
    logging.info("Starting label extraction process")
    extracted_data = extract_labels_from_file(args.input_file, args.audio_dir)
    
    if args.output_dir and args.audio_dir:
        logging.info(f"Writing output files to {args.output_dir}")
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Generate label_descriptors.csv
        unique_labels = get_unique_labels(extracted_data)
        write_label_descriptors(unique_labels, 
                              os.path.join(args.output_dir, "label_descriptors.csv"))
        
        # Write TSV files with audio durations
        write_tsv_files(extracted_data, args.audio_dir, args.output_dir)
        write_label_files(extracted_data, args.output_dir, args.audio_dir)
        logging.info("Process completed successfully")

if __name__ == "__main__":
    main()
