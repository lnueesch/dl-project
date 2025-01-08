import os
import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from util import load_model

# Define parameters
args = {
    'data': './data',  # Path to dataset
    'model': 'experiment/checkpoint.pth.tar',   # Path to model
    'batch_size': 256,
    'workers': 4,  # Number of data loading workers
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

def extract_features(loader, model, dataset_size):
    """Extract features from the dataset."""
    features = None
    labels = []
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(args['device'])
            outputs = model(inputs).cpu().numpy()

            if features is None:
                features = np.zeros((dataset_size, outputs.shape[1]), dtype='float32')

            start_idx = i * args['batch_size']
            end_idx = start_idx + outputs.shape[0]
            features[start_idx:end_idx] = outputs
            labels.extend(targets.numpy())

            if i % 100 == 0:
                print(f"Processed {end_idx}/{dataset_size} samples")

    return features, np.array(labels)

def plot_embeddings(features, labels):
    tsne = TSNE(n_components=2, random_state=0)
    reduced_features = tsne.fit_transform(features)

    plt.figure(figsize=(10, 10))
    for i in range(10):  # Assuming 10 classes for MNIST
        indices = labels == i
        plt.scatter(reduced_features[indices, 0], reduced_features[indices, 1], label=f'Class {i}')
    plt.legend()
    plt.show()

def main():
    # Data preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])

    # Load dataset
    dataset = datasets.MNIST(args['data'], train=False, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=args['batch_size'], shuffle=False, num_workers=args['workers'])

    # Load model
    print(f"=> loading model from {args['model']}")
    model = load_model(args['model'])
    model = model.to(args['device'])
    model.eval()

    # Extract features
    print("Extracting features...")
    features, labels = extract_features(loader, model, len(dataset))

    # Plot feature embeddings
    plot_embeddings(features, labels)

if __name__ == "__main__":
    main()