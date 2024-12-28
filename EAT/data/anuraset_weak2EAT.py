import csv
import argparse
import os

def extract_labels_from_file(input_file):
    """
    Extract labels dynamically from the input CSV file. The function will:
    - Read the species columns from the header (excluding the first two columns).
    - Process each row to extract the filename and present species.
    
    Args:
    - input_file (str): The path to the CSV file containing the data.
    
    Returns:
    - list of tuples: Each tuple contains a filename and a comma-separated string of species labels.
    """
    extracted_data = []
    
    with open(input_file, mode='r', newline='') as file:
        reader = csv.reader(file)
        headers = next(reader)  # Read the header
        
        # Extract species column names (skip the first two columns: MONITORING_SITE, AUDIO_FILE_ID)
        species_columns = headers[2:]
        
        for row in reader:
            filename = row[1]  # The second element is the filename
            labels = []
            
            # Iterate over the species columns (starting from index 2)
            for i, value in enumerate(row[2:], start=0):  # Start at index 0 for species columns
                if value == '2':  # Presence (assuming '2' means present in the dataset)
                    labels.append(species_columns[i])
            
            # Join the labels into a single string, separated by commas
            labels_str = ','.join(labels)
            extracted_data.append((filename, labels_str))
    
    return extracted_data

def get_unique_labels(extracted_data):
    # Get all unique labels
    unique_labels = set()
    for _, labels_str in extracted_data:
        labels = labels_str.split(',')
        unique_labels.update(labels)
    return sorted(list(unique_labels))

def write_label_descriptors(labels, output_path):
    with open(output_path, "w") as f:
        for idx, label in enumerate(labels):
            f.write(f"{label},{idx}\n")

def write_tsv_files(extracted_data, audio_dir, output_dir):
    import soundfile as sf
    
    with open(os.path.join(output_dir, "train.tsv"), "w") as f:
        # Write root directory as first line
        f.write(f"{audio_dir}\n")
        
        cutoff = int(0.8 * len(extracted_data))
        train_data = extracted_data[:cutoff]
        
        # Write audio file paths and their lengths
        for filename, labels in train_data:
            audio_path = os.path.join(audio_dir, filename)
            info = sf.info(audio_path)
            num_samples = int(info.frames)
            rel_path = os.path.relpath(audio_path, audio_dir)
            f.write(f"{rel_path} {num_samples}\n")

    # Same for eval.tsv
    with open(os.path.join(output_dir, "eval.tsv"), "w") as f:
        f.write(f"{audio_dir}\n")
        eval_data = extracted_data[cutoff:]
        for filename, labels in eval_data:
            audio_path = os.path.join(audio_dir, filename)
            info = sf.info(audio_path)
            num_samples = int(info.frames)
            rel_path = os.path.relpath(audio_path, audio_dir)
            f.write(f"{rel_path} {num_samples}\n")

def main():
    parser = argparse.ArgumentParser(description="Extract labels from a CSV file.")
    parser.add_argument("--input_file", required=True, help="Path to the input CSV file.")
    parser.add_argument("--output_dir", help="Directory to store output files.")
    parser.add_argument("--audio_dir", help="Directory containing the audio files.")
    args = parser.parse_args()
    
    extracted_data = extract_labels_from_file(args.input_file)
    
    if args.output_dir and args.audio_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Generate label_descriptors.csv
        unique_labels = get_unique_labels(extracted_data)
        write_label_descriptors(unique_labels, 
                              os.path.join(args.output_dir, "label_descriptors.csv"))
        
        # Write TSV files with audio durations
        write_tsv_files(extracted_data, args.audio_dir, args.output_dir)

if __name__ == "__main__":
    main()
