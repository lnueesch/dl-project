# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import time

# import faiss
# faiss.omp_set_num_threads(1)  # Use a single thread for debugging
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from active_semi_clustering.semi_supervised.pairwise_constraints import PCKMeans
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

ImageFile.LOAD_TRUNCATED_IMAGES = True

__all__ = ['PIC', 'Kmeans', 'PCKmeans', 'cluster_assign', 'arrange_clustering']

def plot_clusters(fig, axes, features, kmeans_labels, true_labels, n_clusters, epoch, save_path=None):
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
    # Reduce to 2D with t-SNE for visualization
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    fraction = 0.05  # Use 10% of the features
    num_samples = int(features.shape[0] * fraction)
    indices = np.random.choice(features.shape[0], num_samples, replace=False)
    reduced_features = tsne.fit_transform(features[indices])
    kmeans_labels = kmeans_labels[indices]
    true_labels = true_labels[indices]

    # Clear previous plots
    for ax in axes:
        ax.clear()

    # Plot K-means clustering
    for cluster in range(n_clusters):
        cluster_indices = (kmeans_labels == cluster)
        axes[0].scatter(reduced_features[cluster_indices, 0],
                        reduced_features[cluster_indices, 1],
                        label=f"Cluster {cluster}", alpha=0.6)
    axes[0].set_title(f"K-means Clusters (Epoch {epoch})")
    axes[0].legend()

    # Plot True labels
    unique_labels = np.unique(true_labels)
    for label in unique_labels:
        label_indices = (true_labels == label)
        axes[1].scatter(reduced_features[label_indices, 0],
                        reduced_features[label_indices, 1],
                        label=f"Label {label}", alpha=0.6)
    axes[1].set_title(f"True Labels (Epoch {epoch})")
    axes[1].legend()

    # Redraw the figure
    fig.tight_layout()
    plt.pause(0.1)

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
    A dataset where the new image labels are given in argument.
    Args:
        image_indexes (list): list of data indexes
        pseudolabels (list): list of labels for each data
        dataset (Dataset or list): original dataset
        transform (callable, optional): a function/transform that takes in
                                        a PIL image or Tensor and returns
                                        a transformed version
    """

    def __init__(self, image_indexes, pseudolabels, dataset, transform=None):
        self.imgs = self.make_dataset(image_indexes, pseudolabels, dataset)
        self.transform = transform

    def make_dataset(self, image_indexes, pseudolabels, dataset):
        label_to_idx = {label: idx for idx, label in enumerate(set(pseudolabels))}
        images = []
        for j, idx in enumerate(image_indexes):
            # Use the image Tensor directly from MNIST or a similar dataset
            img = dataset[idx][0]
            pseudolabel = label_to_idx[pseudolabels[j]]
            images.append((img, pseudolabel))
        return images

    def __getitem__(self, index):
        img, pseudolabel = self.imgs[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, pseudolabel

    def __len__(self):
        return len(self.imgs)


def preprocess_features(npdata, pca=32):
    """Applies PCA-reducing, whitening, and L2-normalization to the data."""
    _, ndim = npdata.shape
    print("ndim: ", ndim)
    npdata = npdata.astype('float32')

    # Check for degenerate data
    if np.any(np.all(npdata == 0, axis=1)):
        raise ValueError("Input data in preprocess_features has all-zero rows.")

    # Check for NaN/infinite
    if np.any(np.isnan(npdata)) or np.any(np.isinf(npdata)):
        raise ValueError("Input data contains NaNs or infinite values.")

    # PCA-whitening with Faiss
    # mat = faiss.PCAMatrix(ndim, pca, eigen_power=-0.5)
    # mat.train(npdata)
    # if not mat.is_trained:
    #     raise ValueError("PCA training failed; check the input data.")
    # npdata = mat.apply_py(npdata)

    # PCA-whitening
    pca_model = PCA(n_components=pca, whiten=True)
    npdata = pca_model.fit_transform(npdata)

    # Check again for NaN/infinite
    if np.any(np.isnan(npdata)) or np.any(np.isinf(npdata)):
        raise ValueError("NaN or Inf detected in preprocessed features.")

    # L2 normalization
    # row_sums = np.linalg.norm(npdata, axis=1)
    # if np.any(row_sums == 0):
    #     raise ValueError("L2 normalization failed due to a zero-row norm.")
    # npdata = npdata / row_sums[:, np.newaxis]

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


# def make_graph(xb, nnn, device):
#     """Builds a graph of nearest neighbors using Faiss, depending on the device."""
#     N, dim = xb.shape
#     if device == 'cuda' and is_gpu_available():
#         print('Using GPU for Faiss')
#         res = faiss.StandardGpuResources()
#         flat_config = faiss.GpuIndexFlatConfig()
#         flat_config.device = int(torch.cuda.current_device())
#         index = faiss.GpuIndexFlatL2(res, dim, flat_config)
#     elif device == 'mps' and is_mps_available():
#         # If MPS is available, attempt MPS usage (if Faiss supports MPS)
#         print('Using MPS for Faiss')
#         res = faiss.StandardGpuResources()
#         flat_config = faiss.GpuIndexFlatConfig()
#         flat_config.device = 0
#         index = faiss.GpuIndexFlatL2(res, dim, flat_config)
#     else:
#         print('Using CPU for Faiss')
#         index = faiss.IndexFlatL2(dim)

#     index.add(xb)
#     D, I = index.search(xb, nnn + 1)
#     return I, D


def cluster_assign(images_lists, dataset):
    """
    Creates a dataset from clustering, with clusters as labels.
    Args:
        images_lists (list of list): for each cluster, the list of image indexes
                                     belonging to this cluster
        dataset (Dataset): the original dataset
    Returns:
        ReassignedDataset(torch.utils.data.Dataset): dataset with cluster labels
    """
    assert images_lists is not None
    pseudolabels = []
    image_indexes = []
    for cluster_idx, images in enumerate(images_lists):
        image_indexes.extend(images)
        pseudolabels.extend([cluster_idx] * len(images))

    # Transform: only normalization for MNIST-like grayscale
    t = transforms.Compose([
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean/std
    ])

    return ReassignedDataset(image_indexes, pseudolabels, dataset, t)


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
    # if device == 'cuda' and is_gpu_available():
    #     clus = faiss.Clustering(d, nmb_clusters)
    #     res = faiss.StandardGpuResources()
    #     flat_config = faiss.GpuIndexFlatConfig()
    #     flat_config.device = int(torch.cuda.current_device())
    #     index = faiss.GpuIndexFlatL2(res, d, flat_config)
    # elif device == 'mps' and is_mps_available():
    #     print('Using MPS (Metal Performance Shaders) for Faiss')
    #     clus = faiss.Clustering(d, nmb_clusters)
    #     res = faiss.StandardGpuResources()
    #     flat_config = faiss.GpuIndexFlatConfig()
    #     flat_config.device = 0
    #     index = faiss.GpuIndexFlatL2(res, d, flat_config)
    # else:
    #     print('Using CPU for Faiss')
    #     clus = faiss.Clustering(d, nmb_clusters)
    #     index = faiss.IndexFlatL2(d)

    # clus.seed = np.random.randint(1234)
    # clus.niter = 50
    # clus.max_points_per_centroid = n_data // 3
    # clus.train(xb, index)
    # _, I = index.search(xb, 1)

    # stats = clus.iteration_stats
    # obj = np.array([stats.at(i).obj for i in range(stats.size())])
    # losses = obj
    # if verbose:
    #     print('k-means loss evolution:', losses)

    # # Print distribution of samples over the centroids
    # unique, counts = np.unique(I, return_counts=True)
    # distribution = dict(zip(unique, counts))
    # print('Distribution of samples over centroids:', distribution)

    # I is shape (N,1), flatten it
    # return [int(n[0]) for n in I], losses[-1]

    # Step 2: Perform KMeans clustering
    print('Perform KMeans from sklearn')
    kmeans = KMeans(n_clusters=nmb_clusters, random_state=42, n_init='auto', max_iter=300)
    kmeans.fit(xb)

    # Retrieve cluster assignments and inertia (loss)
    cluster_assignments = kmeans.labels_
    loss = kmeans.inertia_

    if verbose:
        print('k-means summed loss:', loss)

    # Print distribution of samples over the centroids
    unique, counts = np.unique(cluster_assignments, return_counts=True)
    distribution = dict(zip(unique, counts))
    print('Distribution of samples over centroids:', distribution)

    return cluster_assignments.tolist(), loss/n_data


def run_pckmeans(x, nmb_clusters, pairwise_constraints, verbose=False): 
    """
    Runs pck-means on the given data (x). This function:
      1) Preprocesses (PCA, whiten, L2 norm)
      2) Runs pck-means 
    Args:
        x (np.array): data of shape (N, dim)
        nmb_clusters (int): number of clusters
        pairwise_constraints: tupel with must-link and cannot-link constraints
        verbose (bool): if True, prints progress
    Returns:
        I (list): cluster assignments for each sample
        loss (float): final k-means loss
    """
    # -- FIX: reorder to avoid dimension mismatch --
    # Step 1: Preprocess features
    xb = preprocess_features(x)
    n_data, d = xb.shape  # read dimension AFTER PCA
    print("n_data: ", n_data)
    
    # Step 2: Perform PCKMeans clustering
    print('Perform PCKMeans from sklearn')
    pckmeans = PCKMeans(n_clusters=nmb_clusters)
    pckmeans.fit(xb, ml=pairwise_constraints[0], cl=pairwise_constraints[1])
    
    # Retrieve cluster assignments and inertia (loss)
    cluster_assignments = pckmeans.labels_
    # loss = pckmeans.inertia_
    loss = -1

    # if verbose:
    #     print('pck-means summed loss:', loss)

    # Print distribution of samples over the centroids
    unique, counts = np.unique(cluster_assignments, return_counts=True)
    distribution = dict(zip(unique, counts))
    print('Distribution of samples over centroids:', distribution)

    return cluster_assignments.tolist(), loss


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


class Kmeans(object):
    """
    Simple K-means wrapper class. The `cluster` method:
      - calls `run_kmeans` exactly once
      - stores images_lists for subsequent usage
      - plots clustering results if `plot` flag is set
    """
    def __init__(self, k, device='cpu', plot=True):
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
            save_path = f"cluster_plot_epoch_{epoch}.png"  # Optional save path
            plot_clusters(fig, axes, x_data, kmeans_labels, true_labels, self.k, epoch, save_path)

        return loss

class PCKmeans(object):
    """
    Simple PCK-means wrapper class. The `cluster` method:
      - calls `run_pckmeans` exactly once
      - stores images_lists for subsequent usage
      - plots clustering results if `plot` flag is set
    """
    def __init__(self, k, device='cpu', plot=True, pairwise_constraints=None):
        self.k = k # Number of clusters
        self.device = device # Device to run k-means on
        self.plot = plot  # New flag to enable/disable plotting
        self.pairwise_constraints = pairwise_constraints

    def cluster(self, fig, axes, x_data, true_labels=None, epoch=None, verbose=False):
        """
        Performs pck-means clustering on x_data.
        Args:
            x_data (np.array): Data to cluster of shape (N, dim)
            true_labels (np.array): Ground truth labels (N,)
            epoch (int): Current epoch (for annotation in plots)
            verbose (bool): if True, prints debug info
        """
        end = time.time()

        # Directly call run_pckmeans on x_data
        I, loss = run_pckmeans(x_data, self.k, self.pairwise_constraints, verbose)

        # Build images_lists, a list of lists: for each cluster, the assigned sample indices
        self.images_lists = [[] for _ in range(self.k)]
        for i in range(len(x_data)):
            self.images_lists[I[i]].append(i)

        if verbose:
            elapsed = time.time() - end
            print(f'pck-means time: {elapsed:.0f} s')

        # Plot clusters if enabled
        if self.plot and true_labels is not None and epoch is not None:
            kmeans_labels = np.array(I)  # Cluster assignments
            save_path = f"cluster_plot_epoch_{epoch}.png"  # Optional save path
            plot_clusters(fig, axes, x_data, kmeans_labels, true_labels, self.k, epoch, save_path)

        return loss
    

def make_adjacencyW(I, D, sigma):
    """
    Create adjacency matrix with a Gaussian kernel.
    Args:
        I (ndarray): for each vertex, the indices of its nnn neighbors (including self in col 0)
        D (ndarray): for each vertex, the L2 distances to neighbors (col 0 is self, so distance=0)
        sigma (float): bandwidth of the Gaussian kernel
    Returns:
        csr_matrix: a sparse affinity matrix
    """
    V, k = I.shape
    k = k - 1  # exclude the self column
    indices = np.reshape(np.delete(I, 0, 1), (1, -1))
    indptr = np.multiply(k, np.arange(V + 1))

    def exp_ker(dist):
        return np.exp(-dist / (sigma**2))
    exp_ker = np.vectorize(exp_ker)

    res_D = exp_ker(D)
    data = np.reshape(np.delete(res_D, 0, 1), (1, -1))
    adj_matrix = csr_matrix((data[0], indices[0], indptr), shape=(V, V))
    return adj_matrix


def run_pic(I, D, sigma, alpha):
    """
    Run PIC (Power Iteration Clustering).
    """
    a = make_adjacencyW(I, D, sigma)
    graph = a + a.transpose()
    nim = graph.shape[0]
    W = graph

    v0 = np.ones(nim) / nim
    v = v0.astype('float32')

    for i in range(200):
        vnext = W.transpose().dot(v)
        vnext = alpha * vnext + (1 - alpha) / nim
        vnext /= vnext.sum()
        v = vnext

        if i == 199:  # last iteration
            clust = find_maxima_cluster(W, v)

    return [int(ci) for ci in clust]


def find_maxima_cluster(W, v):
    """
    Helper for PIC that groups each node to the 'best' cluster root.
    """
    n, m = W.shape
    assert n == m
    assign = np.zeros(n, dtype=int)
    pointers = list(range(n))
    for i in range(n):
        best_vi = 0
        l0 = W.indptr[i]
        l1 = W.indptr[i + 1]
        for l in range(l0, l1):
            j = W.indices[l]
            vi = W.data[l] * (v[j] - v[i])
            if vi > best_vi:
                best_vi = vi
                pointers[i] = j

    cluster_ids = -1 * np.ones(n, dtype=int)
    n_clus = 0
    for i in range(n):
        if pointers[i] == i:
            cluster_ids[i] = n_clus
            n_clus += 1

    for i in range(n):
        current_node = i
        while pointers[current_node] != current_node:
            current_node = pointers[current_node]
        assign[i] = cluster_ids[current_node]
        assert assign[i] >= 0

    return assign


# class PIC(object):
#     """
#     Power Iteration Clustering on an nnn-graph with a Gaussian kernel.
#     """

#     def __init__(self, args=None, sigma=0.2, nnn=5, alpha=0.001, distribute_singletons=True):
#         self.sigma = sigma
#         self.alpha = alpha
#         self.nnn = nnn
#         self.distribute_singletons = distribute_singletons

#     def cluster(self, fig, axes, data, verbose=False):
#         end = time.time()

#         # 1) Preprocess data
#         xb = preprocess_features(data)

#         # 2) Construct nearest-neighbor graph
#         I, D = make_graph(xb, self.nnn, device='cpu')  # or device from args

#         # 3) Run PIC
#         clust = run_pic(I, D, self.sigma, self.alpha)
#         images_lists_dict = {}
#         for h in set(clust):
#             images_lists_dict[h] = []
#         for idx, c in enumerate(clust):
#             images_lists_dict[c].append(idx)

#         # 4) Optionally reassign singletons
#         if self.distribute_singletons:
#             clust_NN = {}
#             for i in images_lists_dict:
#                 if len(images_lists_dict[i]) == 1:
#                     s = images_lists_dict[i][0]
#                     for neighbor in I[s, 1:]:
#                         if len(images_lists_dict[clust[neighbor]]) != 1:
#                             clust_NN[s] = neighbor
#                             break
#             for s in clust_NN:
#                 old_cluster = clust[s]
#                 images_lists_dict[old_cluster].remove(s)
#                 clust[s] = clust[clust_NN[s]]
#                 images_lists_dict[clust[s]].append(s)

#         self.images_lists = []
#         for c in images_lists_dict:
#             self.images_lists.append(images_lists_dict[c])

#         if verbose:
#             print('pic time: {0:.0f} s'.format(time.time() - end))
#         return 0