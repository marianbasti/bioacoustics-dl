import csv
import argparse
import os
import logging
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def calculate_split_weights(data, audio_files):
    """Calculate normalized weights for valid audio files only"""
    label_counts = {}
    valid_samples = []
    invalid_count = 0
    weights = []
    
    # First pass - count valid combinations
    for filename, labels_str in data:
        norm_path = os.path.normpath(filename)
        if norm_path not in audio_files:
            invalid_count += 1
            # Add zero weight for invalid samples
            weights.append(0.0)
            continue
            
        label_dict = {
            species: level 
            for label in labels_str.split(',')
            for species, level in [label.split('=')] 
            if level != '0'
        }
        
        key = tuple(sorted(label_dict.items()))
        label_counts[key] = label_counts.get(key, 0) + 1
        valid_samples.append((filename, key))
    
    total_valid = len(valid_samples)
    
    # Calculate weights for valid samples
    for filename, key in valid_samples:
        count = label_counts[key]
        weight = total_valid / (len(label_counts) * count)
        weights.append(weight)
    
    # Normalize weights to sum to 1
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    return weights

def write_weights_file(train_data, audio_files, output_dir):
    """Write normalized weights for training set"""
    weights = calculate_split_weights(train_data, audio_files)
    
    output_path = os.path.join(output_dir, "weight_train_all.csv") 
    logging.info(f"Writing {len(weights)} weights to {output_path}")
    
    np.savetxt(output_path, weights)

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
        headers = next(reader)
        
        species_columns = headers[2:]
        logging.info(f"Found {len(species_columns)} species columns")
        
        row_count = 0
        for row in reader:
            # Handle both nested and flat structures
            if os.path.sep in row[1]:  # If path contains separators, use as is
                filename = f"{row[0]}/{row[1]}.wav"
            else:  # For flat structure, just use the file ID
                filename = f"{row[1]}.wav"
            
            filename = os.path.normpath(filename)
            labels = []
            
            for i, value in enumerate(row[2:], start=0):
                level = value if value in ['1','2','3'] else '0'
                labels.append(f"{species_columns[i]}={level}")
            
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
    
    # Get valid audio files with both full and base names as keys
    audio_files = {}
    for path, samples in collect_audio_files(audio_dir):
        norm_path = os.path.normpath(path)
        audio_files[norm_path] = (path, samples)
        # Also store by basename for flat directory matching
        audio_files[os.path.basename(norm_path)] = (path, samples)
    
    def write_set(data, output_file):
        valid_count = 0
        written_files = set()
        with open(os.path.join(output_dir, output_file), "w") as f:
            for filename, labels in data:
                norm_path = os.path.normpath(filename)
                basename = os.path.basename(norm_path)
                
                # Try both full path and basename matching
                if norm_path in audio_files:
                    path_key = norm_path
                elif basename in audio_files:
                    path_key = basename
                else:
                    continue
                
                rel_path = audio_files[path_key][0]
                file_id = os.path.splitext(rel_path)[0]
                
                level_dict = {}
                for label in labels.split(','):
                    species, level = label.split('=')
                    if level != '0':
                        level_dict[species] = level
                
                level_labels = [f"{species}={level_dict[species]}" 
                              for species in sorted(level_dict.keys())]
                f.write(f"{file_id}\t{' '.join(level_labels)}\n")
                written_files.add(path_key)
                valid_count += 1
                
        return valid_count, written_files
    
    # Process only valid files
    valid_entries = [(f, l) for f, l in extracted_data 
                    if os.path.normpath(f) in audio_files]
    cutoff = int(0.8 * len(valid_entries))
    
    train_valid_count, train_files = write_set(valid_entries[:cutoff], "train.lbl")
    eval_valid_count, eval_files = write_set(valid_entries[cutoff:], "eval.lbl")
    
    logging.info(f"Wrote {train_valid_count} train and {eval_valid_count} eval entries")
    return train_files, eval_files

def write_tsv_files(extracted_data, audio_dir, output_dir, train_files=None, eval_files=None):
    logging.info("Writing TSV files")
    
    audio_files = {}
    for path, samples in collect_audio_files(audio_dir):
        normalized_path = os.path.normpath(path)
        audio_files[normalized_path] = (path, samples)
        # Also store by basename
        audio_files[os.path.basename(normalized_path)] = (path, samples)
    
    def write_set(data, output_file, allowed_files=None):
        valid_entries = []
        written_count = 0
        with open(os.path.join(output_dir, output_file), "w") as f:
            f.write(f"{audio_dir}\n")
            for filename, _ in data:
                norm_path = os.path.normpath(filename)
                basename = os.path.basename(norm_path)
                
                # Try both full path and basename matching
                if norm_path in audio_files and (allowed_files is None or norm_path in allowed_files):
                    path, num_samples = audio_files[norm_path]
                    valid_entries.append((filename, _))
                    f.write(f"{path}\t{num_samples}\n")
                    written_count += 1
                elif basename in audio_files and (allowed_files is None or basename in allowed_files):
                    path, num_samples = audio_files[basename] 
                    valid_entries.append((filename, _))
                    f.write(f"{path}\t{num_samples}\n")
                    written_count += 1
                    
        logging.info(f"Wrote {written_count} entries to {output_file}")
        return valid_entries

    # Split data ensuring only valid entries are used
    all_valid_entries = [(f, l) for f, l in extracted_data if os.path.normpath(f) in audio_files]
    cutoff = int(0.8 * len(all_valid_entries))
    
    train_data = all_valid_entries[:cutoff]
    eval_data = all_valid_entries[cutoff:]
    
    train_valid = write_set(train_data, "train.tsv", train_files)
    eval_valid = write_set(eval_data, "eval.tsv", eval_files)
    
    return train_valid, eval_valid

def main():
    parser = argparse.ArgumentParser(description="Extract labels from a CSV file.")
    parser.add_argument("--input_file", required=True, help="Path to the input CSV file.")
    parser.add_argument("--output_dir", required=True, help="Directory to store output files.")
    parser.add_argument("--audio_dir", required=True, help="Directory containing the audio files.")
    args = parser.parse_args()
    
    logging.info("Starting label extraction process")
    extracted_data = extract_labels_from_file(args.input_file, args.audio_dir)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Write label files first
    train_files, eval_files = write_label_files(extracted_data, args.output_dir, args.audio_dir)
    
    # Then write TSV files using the same file sets
    train_valid, eval_valid = write_tsv_files(extracted_data, args.audio_dir, args.output_dir,
                                            train_files, eval_files)
    
    # Generate label descriptors from valid entries only
    unique_labels = get_unique_labels(train_valid + eval_valid)
    write_label_descriptors(unique_labels, 
                          os.path.join(args.output_dir, "label_descriptors.csv"))
    
    # Calculate weights for training set
    audio_files = {os.path.normpath(path): samples 
                  for path, samples in collect_audio_files(args.audio_dir)}
    write_weights_file(train_valid, audio_files, args.output_dir)
    
    logging.info("Process completed successfully")

if __name__ == "__main__":
    main()
