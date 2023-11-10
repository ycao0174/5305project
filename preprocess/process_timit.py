import deeplake
import json
import os
import soundfile as sf
from tqdm import tqdm

def preprocess_timit_dataset(out_dir):
    """Preprocess TIMIT dataset and create .json files with file information."""
    # Load the datasets
    train_dataset = deeplake.load("hub://activeloop/timit-train")
    test_dataset = deeplake.load("hub://activeloop/timit-test")  # Assuming the URL is similar to train

    # Process each dataset
    for dataset_type, dataset in zip(['train', 'test'], [train_dataset, test_dataset]):
        file_infos = []
        for item in tqdm(dataset):
            # Assuming each item in dataset is a dictionary-like object with 'wav' key
            wav_data = item['wav']
            wav_path = os.path.join(out_dir, dataset_type, item['file_name'])  # Adjust the path as needed
            # Write the WAV data to file
            sf.write(wav_path, wav_data, 16000)  # Replace 16000 with actual sample rate of TIMIT dataset
            file_infos.append((wav_path, len(wav_data)))

        # Create output directory if it does not exist
        dataset_out_dir = os.path.join(out_dir, dataset_type)
        os.makedirs(dataset_out_dir, exist_ok=True)
        # Write file information to JSON
        with open(os.path.join(dataset_out_dir, 'metadata.json'), 'w') as f:
            json.dump(file_infos, f, indent=4)

if __name__ == "__main__":
    out_directory = 'path_to_output_directory'  # Set this to your desired output directory
    preprocess_timit_dataset(out_directory)

