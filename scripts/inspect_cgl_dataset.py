
import sys
import os
from datasets import load_dataset, get_dataset_config_names

# Add src to path so we can load our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def inspect_dataset(dataset_name):
    """
    Loads a dataset from Hugging Face and prints its information.
    """
    try:
        # Get available configurations for the dataset
        configs = get_dataset_config_names(dataset_name)
        print(f"Available configurations for {dataset_name}: {configs}")

        # Load the dataset with the first available configuration
        if configs:
            config_name = configs[0]
            print(f"Loading dataset with configuration: '{config_name}'")
            dataset = load_dataset(dataset_name, name=config_name)
        else:
            print("No specific configurations found. Loading with default.")
            dataset = load_dataset(dataset_name)

        print("\nDataset Information:")
        print(dataset)

        # Print features of the 'train' split
        if 'train' in dataset:
            print("\nFeatures of 'train' split:")
            print(dataset['train'].features)
            print("\nExample of a record:")
            print(dataset['train'][0])

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    cgl_dataset_name = "creative-graphic-design/CGL-Dataset"
    inspect_dataset(cgl_dataset_name)
