import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.cluster import KMeans
import numpy as np

class Preprocessor:
    def __init__(self, batch_size=256):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        self.train_dataset = datasets.MNIST(
            root='./data', train=True, download=True, transform=transform)
        self.test_dataset = datasets.MNIST(
            root='./data', train=False, download=True, transform=transform)

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False)


class CNN(nn.Module):
    def __init__(self, feature_dim=64):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),  # Output: 32x28x28
            nn.ReLU(),
            nn.MaxPool2d(2),  # Output: 32x14x14
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),  # Output: 64x14x14
            nn.ReLU(),
            nn.MaxPool2d(2)   # Output: 64x7x7
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * 7 * 7, feature_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

class DeepCluster:
    def __init__(self, model, n_clusters=10, device='cpu'):
        self.model = model.to(device)
        self.n_clusters = n_clusters
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def extract_features(self, dataloader):
        self.model.eval()
        features = []
        with torch.no_grad():
            for data, _ in dataloader:
                data = data.to(self.device)
                outputs = self.model(data)
                features.append(outputs.cpu().numpy())
        features = np.concatenate(features, axis=0)
        return features

    def cluster(self, features):
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0)
        pseudo_labels = kmeans.fit_predict(features)
        return pseudo_labels

    def create_dataloader_with_pseudo_labels(self, dataset, pseudo_labels, batch_size=256):
        pseudo_labeled_dataset = PseudoLabelDataset(
            dataset, pseudo_labels)
        dataloader = DataLoader(
            pseudo_labeled_dataset, batch_size=batch_size, shuffle=True)
        return dataloader

    def train(self, dataloader, epochs=1):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for data, labels in dataloader:
                data = data.to(self.device)
                labels = labels.to(self.device).long()  # Convert labels to torch.long

                outputs = self.model(data)

                # If output dimension is not matching number of clusters, adjust it
                if outputs.size(1) != self.n_clusters:
                    raise ValueError(f"Output dimension ({outputs.size(1)}) does not match number of clusters ({self.n_clusters}).")

                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}")


class PseudoLabelDataset(Dataset):
    def __init__(self, original_dataset, pseudo_labels):
        self.dataset = original_dataset
        self.pseudo_labels = pseudo_labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, _ = self.dataset[idx]
        pseudo_label = self.pseudo_labels[idx]
        return data, pseudo_label


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize preprocessor and model
    preprocessor = Preprocessor(batch_size=256)
    feature_dim = 64  # Increase feature dimension for better representation
    model = CNN(feature_dim=feature_dim).to(device)

    # Initialize DeepCluster
    n_clusters = 10  # Number of clusters
    deepcluster = DeepCluster(model, n_clusters=n_clusters, device=device)

    # Number of clustering-training iterations
    num_iterations = 5

    for iteration in range(num_iterations):
        print(f"\nIteration {iteration+1}/{num_iterations}")

        # Step 1: Extract features
        features = deepcluster.extract_features(preprocessor.train_loader)
        print("Features extracted.")

        # Step 2: Perform k-means clustering
        pseudo_labels = deepcluster.cluster(features)
        print("K-means clustering done.")

        # Step 3: Create new dataloader with pseudo-labels
        train_loader = deepcluster.create_dataloader_with_pseudo_labels(
            preprocessor.train_dataset, pseudo_labels)
        print("Dataloader with pseudo-labels created.")

        # Step 4: Train the model with pseudo-labels
        epochs_per_iteration = 5  # Train for more epochs per iteration
        deepcluster.train(train_loader, epochs=epochs_per_iteration)

    print("\nTraining completed.")

    # Evaluation (Optional)
    # Since this is an unsupervised method, evaluation can be done by checking clustering accuracy
    # against true labels. This is for demonstration purposes.

    # Extract features from test set
    test_features = deepcluster.extract_features(preprocessor.test_loader)
    # Cluster test features
    test_pseudo_labels = deepcluster.cluster(test_features)
    # Compare with true labels
    true_labels = preprocessor.test_dataset.targets.numpy()
    from sklearn.metrics.cluster import adjusted_rand_score
    ari = adjusted_rand_score(true_labels, test_pseudo_labels)
    print(f"Adjusted Rand Index on test set: {ari:.4f}")

if __name__ == "__main__":
    main()
