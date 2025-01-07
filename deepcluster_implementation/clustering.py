# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import time

import faiss
faiss.omp_set_num_threads(1)  # Use a single thread for debugging
import numpy as np
from PIL import Image
from PIL import ImageFile
from scipy.sparse import csr_matrix, find
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

ImageFile.LOAD_TRUNCATED_IMAGES = True

seed=42
np.random.seed(seed)


__all__ = ['Kmeans', 'PCKmeans', 'cluster_assign', 'arrange_clustering']

def plot_clusters(fig, axes, features, kmeans_labels, true_labels, n_clusters, epoch, save_path=None, mode='TSNE'):
    """
    Update the same figure for cluster visualization during training.
    Args:
        fig: Matplotlib figure object.
        axes: Matplotlib axes array.
        features (np.array): PCA-reduced feature array (N x d).
        kmeans_labels (np.array): Cluster labels from k-means (N,).
        true_labels (np.array): Ground truth labels (N,).
        n_clusters (int): Number of clusters.
        epoch (int): Current epoch for annotation.
    """
    fraction = 0.05 # Fraction of data to visualize
    num_samples = int(features.shape[0] * fraction)
    indices = np.random.choice(features.shape[0], num_samples, replace=False)
    # Reduce to 2D with t-SNE for visualization
    if mode == 'TSNE':
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        reduced_features = tsne.fit_transform(features[indices])
    elif mode == 'PCA':
        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(features[indices])
    else:
        raise ValueError(f"Unknown mode: {mode}")

    kmeans_labels = kmeans_labels[indices]
    true_labels = true_labels[indices]

    # Clear previous plots
    for ax in axes:
        ax.clear()

    # Use LaTeX font
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Plot K-means clustering
    for cluster in range(n_clusters):
        cluster_indices = (kmeans_labels == cluster)
        axes[0].scatter(reduced_features[cluster_indices, 0],
                        reduced_features[cluster_indices, 1],
                        label=f"Cluster {cluster}", alpha=0.6)
    axes[0].set_title(f"(PC)K-means Clusters (Epoch {epoch})")
    axes[0].legend()

    # Plot True labels
    unique_labels = np.unique(true_labels)
    for label in unique_labels:
        label_indices = (true_labels == label)
        axes[1].scatter(
            reduced_features[label_indices, 0],
            reduced_features[label_indices, 1],
            label=f"Label {label}", alpha=0.6
        )
    axes[1].set_title(f"True Labels (Epoch {epoch})")
    axes[1].legend()

    # Redraw the figure
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    plt.pause(1)

def pil_loader(path):
    """Loads an image (for demonstration)."""
    with open(path, 'rb') as f:
        # open the image file. it is a MNIST-like image
        img = Image.open(f)
        img = img.convert('L')
        cv2.imshow('image', np.array(img))
        cv2.waitKey(0)
        return img


class ReassignedDataset(data.Dataset):
    """
    A dataset where the original data is kept but with new labels.
    """
    def __init__(self, image_indexes, pseudolabels, dataset):
        """
        Args:
            image_indexes (list): list of data indexes
            pseudolabels (list): list of labels for each data point
            dataset (Dataset): original dataset
        """
        self.dataset = dataset
        self.image_indexes = image_indexes
        # Create a mapping from label to index for efficiency
        unique_labels = set(pseudolabels)
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(unique_labels))}
        self.pseudolabels = [self.label_to_idx[label] for label in pseudolabels]

    def __getitem__(self, index):
        """
        Returns:
            data: original data
            pseudolabel: cluster assignment (0 to k-1)
        """
        img, _ = self.dataset[self.image_indexes[index]]
        return img, self.pseudolabels[index]

    def __len__(self):
        return len(self.image_indexes)


def preprocess_features(npdata, pca=64):
    """Applies PCA-reducing, whitening, and L2-normalization to the data."""
    _, ndim = npdata.shape
    print("in_dim: ", ndim)
    print("out_dim: ", pca)
    npdata = npdata.astype('float32')

    # Check for degenerate data
    if np.any(np.all(npdata == 0, axis=1)):
        raise ValueError("Input data in preprocess_features has all-zero rows.")

    # Check for NaN/infinite
    if np.any(np.isnan(npdata)) or np.any(np.isinf(npdata)):
        raise ValueError("Input data contains NaNs or infinite values.")

    # PCA-whitening with Faiss
    mat = faiss.PCAMatrix(ndim, pca, eigen_power=-0.5)
    mat.train(npdata)
    if not mat.is_trained:
        raise ValueError("PCA training failed; check the input data.")
    npdata = mat.apply_py(npdata)

    # Check again for NaN/infinite
    if np.any(np.isnan(npdata)) or np.any(np.isinf(npdata)):
        raise ValueError("NaN or Inf detected in preprocessed features.")

    # L2 normalization
    row_sums = np.linalg.norm(npdata, axis=1)
    if np.any(row_sums == 0):
        raise ValueError("L2 normalization failed due to a zero-row norm.")
    npdata = npdata / row_sums[:, np.newaxis]

    # Final check
    if np.any(np.isnan(npdata)) or np.any(np.isinf(npdata)):
        raise ValueError("NaN or Inf detected after final normalization.")

    return npdata


def is_gpu_available():
    """Check if CUDA is available."""
    return torch.cuda.is_available()

def is_mps_available():
    """Check if MPS (Metal) is available on Mac."""
    return torch.backends.mps.is_available()


def make_graph(xb, nnn, device):
    """Builds a graph of nearest neighbors using Faiss, depending on the device."""
    N, dim = xb.shape
    if device == 'cuda' and is_gpu_available():
        print('Using GPU for Faiss')
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.device = int(torch.cuda.current_device())
        index = faiss.GpuIndexFlatL2(res, dim, flat_config)
    elif device == 'mps' and is_mps_available():
        # If MPS is available, attempt MPS usage (if Faiss supports MPS)
        print('Using MPS for Faiss')
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.device = 0
        index = faiss.GpuIndexFlatL2(res, dim, flat_config)
    else:
        print('Using CPU for Faiss')
        index = faiss.IndexFlatL2(dim)

    index.add(xb)
    D, I = index.search(xb, nnn + 1)
    return I, D

class Kmeans(object):
    """
    Simple K-means wrapper class. The `cluster` method:
      - calls `run_kmeans` exactly once
      - stores images_lists for subsequent usage
      - plots clustering results if `plot` flag is set
    """
    def __init__(self, k, device='cpu', plot=True, constraints=None, labeled_indices=None):
        self.k = k # Number of clusters
        self.device = device # Device to run k-means on
        self.plot = plot  # New flag to enable/disable plotting

    def cluster(self, fig, axes, x_data, true_labels=None, epoch=None, verbose=False):
        """
        Performs k-means clustering on x_data.
        Args:
            x_data (np.array): Data to cluster of shape (N, dim)
            true_labels (np.array): Ground truth labels (N,)
            epoch (int): Current epoch (for annotation in plots)
            verbose (bool): if True, prints debug info
        """
        end = time.time()

        # Directly call run_kmeans on x_data
        I, loss = run_kmeans(x_data, self.k, verbose, self.device)

        # Build images_lists, a list of lists: for each cluster, the assigned sample indices
        self.images_lists = [[] for _ in range(self.k)]
        for i in range(len(x_data)):
            self.images_lists[I[i]].append(i)

        if verbose:
            elapsed = time.time() - end
            print(f'k-means time: {elapsed:.0f} s')

        # Plot clusters if enabled
        if self.plot and true_labels is not None and epoch is not None:
            kmeans_labels = np.array(I)  # Cluster assignments
            # save_path = f"cluster_plot_epoch_{epoch}.png"  # Optional save path
            plot_clusters(fig, axes, x_data, kmeans_labels, true_labels, self.k, epoch)

        return loss

def cluster_assign(images_lists, dataset):
    """
    Creates a dataset from clustering, with clusters as labels.
    Args:
        images_lists (list of list): for each cluster, the list of image indexes
                                    belonging to this cluster
        dataset (Dataset): the original dataset
    Returns:
        ReassignedDataset: dataset with pseudo-labels from clustering
    """
    assert images_lists is not None
    pseudolabels = []
    image_indexes = []
    for cluster_idx, images in enumerate(images_lists):
        image_indexes.extend(images)
        pseudolabels.extend([cluster_idx] * len(images))

    # No transform needed - we'll use the original data as is
    # since it should already be normalized
    return ReassignedDataset(image_indexes, pseudolabels, dataset)


def run_kmeans(x, nmb_clusters, verbose=False, device='cpu'):
    """
    Runs k-means on the given data (x). This function:
      1) Preprocesses (PCA, whiten, L2 norm)
      2) Runs k-means via Faiss
    Args:
        x (np.array): data of shape (N, dim)
        nmb_clusters (int): number of clusters
        verbose (bool): if True, prints progress
        device (str): 'cpu', 'cuda', or 'mps'
    Returns:
        I (list): cluster assignments for each sample
        loss (float): final k-means loss
    """
    # -- FIX: reorder to avoid dimension mismatch --
    # Step 1: Preprocess features
    xb = preprocess_features(x)
    n_data, d = xb.shape  # read dimension AFTER PCA
    print("n_data: ", n_data)

    # Step 2: Perform Faiss k-means
    if device == 'cuda' and is_gpu_available():
        clus = faiss.Clustering(d, nmb_clusters)
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.device = int(torch.cuda.current_device())
        index = faiss.GpuIndexFlatL2(res, d, flat_config)
    elif device == 'mps' and is_mps_available():
        print('Using MPS (Metal Performance Shaders) for Faiss')
        clus = faiss.Clustering(d, nmb_clusters)
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.device = 0
        index = faiss.GpuIndexFlatL2(res, d, flat_config)
    else:
        print('Using CPU for Faiss')
        clus = faiss.Clustering(d, nmb_clusters)
        index = faiss.IndexFlatL2(d)

    clus.seed = np.random.randint(1234)
    clus.niter = 50
    clus.max_points_per_centroid = n_data // 3
    clus.train(xb, index)
    _, I = index.search(xb, 1)

    stats = clus.iteration_stats
    obj = np.array([stats.at(i).obj for i in range(stats.size())])
    losses = obj
    if verbose:
        print('k-means loss evolution:', losses)

    # Print distribution of samples over the centroids
    unique, counts = np.unique(I, return_counts=True)
    distribution = dict(zip(unique, counts))
    print('Distribution of samples over centroids:', distribution)

    # I is shape (N,1), flatten it
    return [int(n[0]) for n in I], losses[-1]


def arrange_clustering(images_lists):
    """
    Convert cluster assignments (images_lists) into a single array aligned with the
    original indexing of the data.
    """
    pseudolabels = []
    image_indexes = []
    for cluster, images in enumerate(images_lists):
        image_indexes.extend(images)
        pseudolabels.extend([cluster] * len(images))
    indexes = np.argsort(image_indexes)
    return np.asarray(pseudolabels)[indexes]


class PCKmeans(object):
    """
    K-means clustering with constraints. The `cluster` method:
      - calls `run_kmeans` exactly once
      - stores images_lists for subsequent usage
      - plots clustering results if `plot` flag is set
    """
    def __init__(self, k, max_iter=10, w=1, device='cpu', plot=True, constraints=None, labeled_indices=None):
        self.n_clusters = k
        self.max_iter = max_iter
        self.w = w
        self.device = device
        self.plot = plot
        self.constraints = constraints
        self.labeled_indices = labeled_indices

    def cluster(self, fig, axes, X, true_labels=None, epoch=None, verbose=False, save_path=None):
        """
        Performs k-means clustering on x_data.
        Args:
            X (np.array): Data to cluster of shape (N, dim)
            true_labels (np.array): Ground truth labels (N,)
            epoch (int): Current epoch (for annotation in plots)
            verbose (bool): if True, prints debug info
        """
        end = time.time()

        X = preprocess_features(X)

        ml, cl = self.constraints

        ml_graph, cl_graph, neighborhoods = preprocess_constraints(ml, cl, X.shape[0])

        cluster_centers = self._initialize_cluster_centers(X, neighborhoods)

        # Repeat until convergence
        for iteration in range(self.max_iter):
            
            # Assign clusters
            labels = self._assign_clusters(X, cluster_centers, ml_graph, cl_graph, self.w)

            # Estimate means
            prev_cluster_centers = cluster_centers
            cluster_centers = self._get_cluster_centers(X, labels)

            # Check for convergence
            difference = (prev_cluster_centers - cluster_centers)
            converged = np.allclose(difference, np.zeros(cluster_centers.shape), atol=2e-4, rtol=0.1)

            print(f"PCKmeans Iteration {iteration + 1}/{self.max_iter} with max diff {np.max(difference)}")
            if converged: break

        self.cluster_centers_, self.labels_ = cluster_centers, labels

        self.images_lists = [[] for _ in range(self.n_clusters)]
        for i in range(len(X)):
            self.images_lists[self.labels_[i]].append(i)

        if self.plot:
            kmeans_labels = np.array(labels)
            plot_clusters(fig, axes, X, kmeans_labels, true_labels, self.n_clusters, epoch, mode='TSNE', save_path=save_path)

        loss = 0

        return loss

    def _initialize_cluster_centers(self, X, neighborhoods):
        neighborhood_centers = np.array([X[neighborhood].mean(axis=0) for neighborhood in neighborhoods])
        neighborhood_sizes = np.array([len(neighborhood) for neighborhood in neighborhoods])

        if len(neighborhoods) > self.n_clusters:
            # Select K largest neighborhoods' centroids
            cluster_centers = neighborhood_centers[np.argsort(neighborhood_sizes)[-self.n_clusters:]]
        else:
            if len(neighborhoods) > 0:
                cluster_centers = neighborhood_centers
            else:
                cluster_centers = np.empty((0, X.shape[1]))

            # FIXME look for a point that is connected by cannot-links to every neighborhood set

            if len(neighborhoods) < self.n_clusters:
                remaining_cluster_centers = X[np.random.choice(X.shape[0], self.n_clusters - len(neighborhoods), replace=False), :]
                cluster_centers = np.concatenate([cluster_centers, remaining_cluster_centers])

        return cluster_centers
    
    def _objective_function(self, X, x_i, centroids, c_i, labels, ml_graph, cl_graph, w):
        distance = 1 / 2 * np.sum((X[x_i] - centroids[c_i]) ** 2)

        ml_penalty = 0
        for y_i in ml_graph[x_i]:
            if labels[y_i] != -1 and labels[y_i] != c_i:
                ml_penalty += w

        cl_penalty = 0
        for y_i in cl_graph[x_i]:
            if labels[y_i] == c_i:
                cl_penalty += w

        return distance + ml_penalty + cl_penalty

    def _assign_clusters(self, X, cluster_centers, ml_graph, cl_graph, w):
        labels = np.full(X.shape[0], fill_value=-1)

        index = list(range(X.shape[0]))
        np.random.shuffle(index)
        for x_i in index:
            labels[x_i] = np.argmin([self._objective_function(X, x_i, cluster_centers, c_i, labels, ml_graph, cl_graph, w) for c_i in range(self.n_clusters)])

        # Handle empty clusters
        # See https://github.com/scikit-learn/scikit-learn/blob/0.19.1/sklearn/cluster/_k_means.pyx#L309
        n_samples_in_cluster = np.bincount(labels, minlength=self.n_clusters)
        empty_clusters = np.where(n_samples_in_cluster == 0)[0]

        if len(empty_clusters) > 0:
            # print("Empty clusters")
            raise ValueError("Empty clusters")

        return labels

    def _get_cluster_centers(self, X, labels):
        return np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])



# Taken from https://github.com/Behrouz-Babaki/COP-Kmeans/blob/master/copkmeans/cop_kmeans.py
def preprocess_constraints(ml, cl, n):
    """Create a graph of constraints for both must- and cannot-links with memory optimization."""
    
    # Use sets for more efficient membership testing and memory usage
    ml_graph = {i: set() for i in range(n)}
    cl_graph = {i: set() for i in range(n)}

    # Process must-link constraints
    for (i, j) in ml:
        ml_graph[i].add(j)
        ml_graph[j].add(i)

    # Find connected components (neighborhoods) using efficient union-find
    def find(parent, i):
        if parent[i] != i:
            parent[i] = find(parent, parent[i])
        return parent[i]

    def union(parent, rank, i, j):
        root_i = find(parent, i)
        root_j = find(parent, j)
        if root_i != root_j:
            if rank[root_i] < rank[root_j]:
                root_i, root_j = root_j, root_i
            parent[root_j] = root_i
            if rank[root_i] == rank[root_j]:
                rank[root_i] += 1

    # Initialize union-find data structures
    parent = list(range(n))
    rank = [0] * n

    # Build connected components
    for (i, j) in ml:
        union(parent, rank, i, j)

    # Create neighborhoods (connected components)
    component_map = {}
    neighborhoods = []
    for i in range(n):
        root = find(parent, i)
        if root not in component_map:
            component_map[root] = len(neighborhoods)
            neighborhoods.append([])
        neighborhoods[component_map[root]].append(i)

    # Process cannot-link constraints more efficiently
    for (i, j) in cl:
        root_i = find(parent, i)
        root_j = find(parent, j)
        
        # Add cannot-links between components
        if root_i != root_j:
            comp_i = component_map[root_i]
            comp_j = component_map[root_j]
            
            # Only process a limited number of nodes from each component
            max_nodes = 1000  # Limit the number of nodes to process
            nodes_i = neighborhoods[comp_i][:max_nodes]
            nodes_j = neighborhoods[comp_j][:max_nodes]
            
            for x in nodes_i:
                for y in nodes_j:
                    cl_graph[x].add(y)
                    cl_graph[y].add(x)

    # Verify constraint consistency
    for i in range(n):
        root_i = find(parent, i)
        for j in cl_graph[i]:
            if find(parent, j) == root_i:
                raise ValueError("Inconsistent constraints detected")

    return ml_graph, cl_graph, neighborhoods