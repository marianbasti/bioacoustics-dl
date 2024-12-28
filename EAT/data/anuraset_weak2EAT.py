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
            f.write(f"{idx},{label}\n")

def main():
    parser = argparse.ArgumentParser(description="Extract labels from a CSV file.")
    parser.add_argument("--input_file", required=True, help="Path to the input CSV file.")
    parser.add_argument("--output_dir", help="Directory to store output files.")
    args = parser.parse_args()
    
    extracted_data = extract_labels_from_file(args.input_file)
    
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Generate label_descriptors.csv
        unique_labels = get_unique_labels(extracted_data)
        write_label_descriptors(unique_labels, 
                              os.path.join(args.output_dir, "label_descriptors.csv"))
        
        # Original code for train/eval split
        cutoff = int(0.8 * len(extracted_data))
        train_data = extracted_data[:cutoff]
        eval_data = extracted_data[cutoff:]
        
        with open(os.path.join(args.output_dir, "train.lbl"), "w") as f:
            for filename, labels in train_data:
                f.write(f"{filename} {labels}\n")  # Changed comma to space
        with open(os.path.join(args.output_dir, "eval.lbl"), "w") as f:
            for filename, labels in eval_data:
                f.write(f"{filename} {labels}\n")  # Changed comma to space
    else:
        for filename, labels in extracted_data:
            print(f"{filename},{labels}")

if __name__ == "__main__":
    main()
