import os
import gzip
import numpy as np
import requests
from scipy.io import savemat

def download_and_extract_mnist():
    base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mnist-mld/"
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz"
    }
    
    def download_file(url, dest):
        response = requests.get(url, stream=True)
        with open(dest, 'wb') as f:
            for chunk in response.iter_content(chunk_size=128):
                f.write(chunk)
    
    def extract_gz(file_path):
        with gzip.open(file_path, 'rb') as f:
            return f.read()
    
    def load_images(file_path):
        with gzip.open(file_path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        return data.reshape(-1, 28, 28) / 255.0
    
    def load_labels(file_path):
        with gzip.open(file_path, 'rb') as f:
            return np.frombuffer(f.read(), np.uint8, offset=8)
    
    # Download and extract files
    for key, filename in files.items():
        if not os.path.exists(filename):
            download_file(base_url + filename, filename)
    
    # Load datasets
    x_train = load_images(files["train_images"])
    y_train = load_labels(files["train_labels"])
    x_test = load_images(files["test_images"])
    y_test = load_labels(files["test_labels"])
    
    # Save datasets to .mat files
    savemat("mnist_train.mat", {"x_train": x_train, "y_train": y_train})
    savemat("mnist_test.mat", {"x_test": x_test, "y_test": y_test})
    
    return (x_train, y_train), (x_test, y_test)

# Example usage
(x_train, y_train), (x_test, y_test) = download_and_extract_mnist()
print(f"Training data shape: {x_train.shape}, Training labels shape: {y_train.shape}")
print(f"Test data shape: {x_test.shape}, Test labels shape: {y_test.shape}")