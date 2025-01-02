# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.

import os
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from util import load_model, AverageMeter

# Define parameters directly
args = {
    'data': './data',  # Path to dataset
    'model': 'experiment/checkpoint.pth.tar',  # Path to model
    'batch_size': 256,
    'workers': 2,
    'verbose': True,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'pca_components': 64,  # Number of PCA components
    'top_k': 5,  # Number of nearest neighbors to retrieve
    'num_queries': 5  # Number of query images to evaluate
}


def main():
    # Data preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])

    # Load dataset
    train_dataset = datasets.MNIST(args['data'], train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST(args['data'], train=False, transform=transform)

    # Use 1/10 of the data for faster processing
    train_dataset = torch.utils.data.Subset(train_dataset, np.arange(0, len(train_dataset), 10))
    val_dataset = torch.utils.data.Subset(val_dataset, np.arange(0, len(val_dataset), 10))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=args['workers'])
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=args['workers'])

    # Load model
    print(f"=> loading model from {args['model']}")
    model = load_model(args['model'])
    model = model.to(args['device'])
    model.eval()

    # Extract features
    print("Extracting features...")
    train_features, train_labels = extract_features(train_loader, model, len(train_dataset))
    val_features, val_labels = extract_features(val_loader, model, len(val_dataset))

    # Apply PCA whitening
    print(f"Applying PCA whitening with {args['pca_components']} components...")
    train_features, pca_model = apply_pca(train_features, args['pca_components'])
    val_features = pca_model.transform(val_features)

    # Normalize features
    train_features = normalize_features(train_features)
    val_features = normalize_features(val_features)

    # Evaluate retrieval
    print(f"Evaluating retrieval for {args['num_queries']} queries...")
    evaluate_retrieval(train_features, train_labels, val_features, val_labels, args['top_k'], train_dataset, val_dataset)


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

            if args['verbose'] and i % 100 == 0:
                print(f"Processed {end_idx}/{dataset_size} samples")

    return features, np.array(labels)


def apply_pca(features, n_components):
    """Fit and apply PCA whitening."""
    # Adjust n_components to ensure it does not exceed the feature dimensions
    n_components = min(n_components, features.shape[1])
    print(f"Adjusted PCA components to: {n_components}")
    
    pca = PCA(n_components=n_components, whiten=True)
    transformed_features = pca.fit_transform(features)
    return transformed_features, pca


def normalize_features(features):
    """L2-normalize features."""
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / (norms + 1e-10)


def evaluate_retrieval(train_features, train_labels, val_features, val_labels, top_k, train_dataset, val_dataset):
    """Evaluate retrieval performance and display retrieved images."""
    # Randomly select queries if num_queries is set
    query_indices = np.random.choice(len(val_labels), min(args['num_queries'], len(val_labels)), replace=False)
    val_features = val_features[query_indices]
    val_labels = val_labels[query_indices]

    similarity = cosine_similarity(val_features, train_features)
    top_k_indices = np.argsort(similarity, axis=1)[:, -top_k:][:, ::-1]

    correct = 0
    for i, indices in enumerate(top_k_indices):
        retrieved_labels = train_labels[indices]
        query_label = val_labels[i]

        # Check if the true label is in the retrieved labels
        if query_label in retrieved_labels:
            correct += 1

        # Plot query and retrieved images
        query_image, _ = val_dataset[query_indices[i]]
        retrieved_images = [train_dataset[idx][0] for idx in indices]

        # Create the plot
        fig, axes = plt.subplots(1, top_k + 1, figsize=(12, 4))
        axes[0].imshow(query_image.squeeze(), cmap="gray")
        axes[0].set_title("Query")
        axes[0].axis("off")

        for j, img in enumerate(retrieved_images):
            axes[j + 1].imshow(img.squeeze(), cmap="gray")
            axes[j + 1].set_title(f"Retrieved {j + 1}")
            axes[j + 1].axis("off")

        plt.tight_layout()
        plt.show()

    # Compute and print accuracy
    accuracy = correct / len(val_labels)
    print(f"Top-{top_k} retrieval accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()