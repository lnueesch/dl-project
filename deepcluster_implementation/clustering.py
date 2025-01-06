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
        axes[1].scatter(
            reduced_features[label_indices, 0],
            reduced_features[label_indices, 1],
            label=f"Label {label}", alpha=0.6
        )
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
    mat = faiss.PCAMatrix(ndim, pca, eigen_power=-0.5)
    mat.train(npdata)
    if not mat.is_trained:
        raise ValueError("PCA training failed; check the input data.")
    npdata = mat.apply_py(npdata)

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


class Kmeans(object):
    """
    Simple K-means wrapper class. The `cluster` method:
      - calls `run_kmeans` exactly once
      - stores images_lists for subsequent usage
      - plots clustering results if `plot` flag is set
    """
    def __init__(self, k, device='cpu', plot=True, constraints=None):
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


class PIC(object):
    """
    Power Iteration Clustering on an nnn-graph with a Gaussian kernel.
    """

    def __init__(self, args=None, sigma=0.2, nnn=5, alpha=0.001, distribute_singletons=True):
        self.sigma = sigma
        self.alpha = alpha
        self.nnn = nnn
        self.distribute_singletons = distribute_singletons

    def cluster(self, fig, axes, data, verbose=False):
        end = time.time()

        # 1) Preprocess data
        xb = preprocess_features(data)

        # 2) Construct nearest-neighbor graph
        I, D = make_graph(xb, self.nnn, device='cpu')  # or device from args

        # 3) Run PIC
        clust = run_pic(I, D, self.sigma, self.alpha)
        images_lists_dict = {}
        for h in set(clust):
            images_lists_dict[h] = []
        for idx, c in enumerate(clust):
            images_lists_dict[c].append(idx)

        # 4) Optionally reassign singletons
        if self.distribute_singletons:
            clust_NN = {}
            for i in images_lists_dict:
                if len(images_lists_dict[i]) == 1:
                    s = images_lists_dict[i][0]
                    for neighbor in I[s, 1:]:
                        if len(images_lists_dict[clust[neighbor]]) != 1:
                            clust_NN[s] = neighbor
                            break
            for s in clust_NN:
                old_cluster = clust[s]
                images_lists_dict[old_cluster].remove(s)
                clust[s] = clust[clust_NN[s]]
                images_lists_dict[clust[s]].append(s)

        self.images_lists = []
        for c in images_lists_dict:
            self.images_lists.append(images_lists_dict[c])

        if verbose:
            print('pic time: {0:.0f} s'.format(time.time() - end))
        return 0
    

import numpy as np
import time
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch

class PCKmeans:
    """
    A faster Pairwise Constrained K-means (PCKmeans) implementation:
      1) Must-link: merged via Union-Find into 'supernodes'.
      2) Cannot-link: penalize supernodes that violate constraints by 
         adding a cost for assigning them to the same cluster.
      3) Vectorized 'batch' updates each iteration for speed.
    """

    def __init__(self, k, device='cpu', plot=True, constraints=None, labeled_indices=None, max_iter=20, 
                 penalty_weight=1.0, must_link_weight=10.0):  # Added must_link_weight
        """
        Args:
            k (int): number of clusters
            device (str): unused here, but kept for compatibility
            plot (bool): if True, calls plot_clusters(...) after clustering
            constraints (dict): {'must_link': [...], 'cannot_link': [...]}
            max_iter (int): maximum number of clustering iterations
            penalty_weight (float): cost added for each cannot-link violation
            must_link_weight (float): cost added for each must-link violation
        """
        self.k = k
        self.device = device
        self.plot = plot
        self.constraints = constraints if constraints is not None else {}
        self.labeled_indices = labeled_indices if labeled_indices is not None else []
        self.max_iter = max_iter
        self.penalty_weight = penalty_weight
        self.must_link_weight = must_link_weight  # Much higher than penalty_weight to enforce must-links

        # Placeholders for final results
        self.images_lists = []
        self.cluster_centers_ = None
        self.labels_ = None
        self.convergence_thresh = 1e-4
        self.constraint_weight = 1.0

    def _kmeans_plus_plus_init(self, x_data, k):
        """Initialize cluster centers using k-means++ algorithm."""
        n_samples, dim = x_data.shape
        centers = np.zeros((k, dim))
        
        # Choose first center randomly
        first_center = x_data[np.random.choice(n_samples)]
        centers[0] = first_center
        
        # Choose remaining centers
        for i in range(1, k):
            # Compute distances to existing centers
            min_dists = np.min([np.linalg.norm(x_data - c, axis=1)**2 for c in centers[:i]], axis=0)
            # Choose next center probabilistically
            probs = min_dists / min_dists.sum()
            next_center_idx = np.random.choice(n_samples, p=probs)
            centers[i] = x_data[next_center_idx]
            
        return centers

    def _compute_transitive_closure(self, must_link_dict):
        """Compute transitive closure of must-link constraints."""
        closure = must_link_dict.copy()
        
        for i in closure:
            stack = list(closure[i])
            while stack:
                j = stack.pop()
                for k in must_link_dict.get(j, set()):
                    if k not in closure[i]:
                        closure[i].add(k)
                        stack.append(k)
        return closure
    def _compute_constraint_violations(self, assignments):
        """Compute constraint violations for current assignments."""
        must_link = self.constraints.get('must_link', [])
        cannot_link = self.constraints.get('cannot_link', [])
        
        must_link_violations = []
        cannot_link_violations = []
        
        for i, j in must_link:
            if assignments[i] != assignments[j]:
                must_link_violations.append((i, j))
                
        for i, j in cannot_link:
            if assignments[i] == assignments[j]:
                cannot_link_violations.append((i, j))
        
        return must_link_violations, cannot_link_violations

    def cluster(self, fig, axes, x_data, true_labels=None, epoch=None, verbose=False):
        """
        Perform semi-supervised clustering (PCKmeans) on x_data.

        Args:
            fig, axes: for plotting (if self.plot is True).
            x_data (np.ndarray or torch.Tensor): shape (N, D) input data
            true_labels (np.ndarray): optional ground-truth labels for plotting
            epoch (int): optional, for plot annotations
            verbose (bool): if True, prints debug info

        Returns:
            float: final sum-of-squared-distances (pseudo-loss)
        """
        start_time = time.time()

        # Convert to numpy if needed
        if isinstance(x_data, torch.Tensor):
            x_data = x_data.cpu().numpy()
        x_data = x_data.astype(np.float32, copy=False)  # ensure float32

        n_samples, dim = x_data.shape
        must_link = self.constraints.get('must_link', [])
        cannot_link = self.constraints.get('cannot_link', [])

        # ---------------------------------------------------------------------
        # 1) Union-Find for Must-Link => build supernodes
        # ---------------------------------------------------------------------
        parent = list(range(n_samples))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a, b):
            rootA, rootB = find(a), find(b)
            if rootA != rootB:
                parent[rootB] = rootA

        # Merge must-link pairs
        for (i, j) in must_link:
            union(i, j)

        # Group by representative
        comp_map = defaultdict(list)
        for i in range(n_samples):
            comp_map[find(i)].append(i)

        # supernodes = list of representative IDs
        supernodes = list(comp_map.keys())
        supernode_count = len(supernodes)

        # For each supernode, gather indices and compute centroid
        supernode_indices = []
        supernode_centroids = np.zeros((supernode_count, dim), dtype=np.float32)

        for s_idx, rep in enumerate(supernodes):
            members = comp_map[rep]
            supernode_indices.append(members)
            supernode_centroids[s_idx] = np.mean(x_data[members], axis=0)

        # ---------------------------------------------------------------------
        # 2) Build cannot-link adjacency in terms of supernode indices
        # ---------------------------------------------------------------------
        # We'll map representative -> index in supernodes just once to avoid .index() overhead
        rep_idx_map = {rep: idx for idx, rep in enumerate(supernodes)}

        cannot_link_idx = [set() for _ in range(supernode_count)]
        for (i, j) in cannot_link:
            sA = find(i)
            sB = find(j)
            # If they happen to be in the same supernode, that is contradictory,
            # but we'll proceed anyway. There's no feasible perfect solution then.
            if sA != sB:
                sA_idx = rep_idx_map[sA]
                sB_idx = rep_idx_map[sB]
                cannot_link_idx[sA_idx].add(sB_idx)
                cannot_link_idx[sB_idx].add(sA_idx)

        # Before main loop, build must-link connections between supernodes
        supernode_must_link = [set() for _ in range(supernode_count)]
        for (i, j) in must_link:
            sA = find(i)
            sB = find(j)
            if sA != sB:  # If they're not already in same supernode
                sA_idx = rep_idx_map[sA]
                sB_idx = rep_idx_map[sB]
                supernode_must_link[sA_idx].add(sB_idx)
                supernode_must_link[sB_idx].add(sA_idx)

        # ---------------------------------------------------------------------
        # 3) K-means style iterative approach (batch updates)
        # ---------------------------------------------------------------------
        rng = np.random.default_rng()

        # Initialize cluster centers from random supernodes (or replicate if fewer supernodes than k)
        if supernode_count <= self.k:
            chosen = list(range(supernode_count))
            while len(chosen) < self.k:
                chosen.append(rng.integers(supernode_count))
        else:
            chosen = rng.choice(supernode_count, size=self.k, replace=False)

        cluster_centers = supernode_centroids[chosen].copy()  # shape (k, dim)
        assignments = np.zeros(supernode_count, dtype=int)

        for iteration in range(self.max_iter):
            # -----------------------------------------------------------------
            # (A) Compute distance matrix from each supernode to each cluster center
            #     dist_matrix[s_idx, c] = 0.5 * ||centroid_s - cluster_centers[c]||^2
            # We do 0.5 * sum of squares because that matches the cost used in standard k-means
            # but we can also drop the 0.5 scaling if we prefer.
            # -----------------------------------------------------------------
            # Expand dims for broadcasting: supernode_centroids (sn, 1, dim) - cluster_centers (1, k, dim)
            diffs = supernode_centroids[:, None, :] - cluster_centers[None, :, :]  # shape (sn, k, dim)
            dist_matrix = 0.5 * np.sum(diffs * diffs, axis=2)                      # shape (sn, k)

            # -----------------------------------------------------------------
            # (B) Build penalty matrix for cannot-link
            #     penalty_matrix[s_idx, c] = (penalty_weight) * (# of neighbors assigned to c)
            # -----------------------------------------------------------------
            penalty_matrix = np.zeros((supernode_count, self.k), dtype=np.float32)

            # We'll go through each supernode sB, read which cluster it's in, say cB,
            # then for each neighbor sA in cannot_link_idx[sB], we add penalty_weight
            # to penalty_matrix[sA, cB] because if sA is also assigned to cB, that is a violation.
            # Then in the assignment step, if sA picks cB, it pays that penalty.
            # => This is a standard "batch" approach: we compute penalty with the *current* assignment
            # and then do an argmin for each supernode simultaneously.
            for sB_idx in range(supernode_count):
                cB = assignments[sB_idx]
                # for each neighbor sA_idx that cannot link with sB_idx
                for sA_idx in cannot_link_idx[sB_idx]:
                    penalty_matrix[sA_idx, cB] += self.penalty_weight

            # Must-link penalties (new)
            for sA_idx in range(supernode_count):
                cA = assignments[sA_idx]
                # For each must-linked supernode
                for sB_idx in supernode_must_link[sA_idx]:
                    # Add high penalty for assigning to different clusters
                    # penalty_matrix[sA_idx] is a vector of penalties for each cluster
                    penalty_matrix[sA_idx] += self.must_link_weight * (np.arange(self.k) != cA)

            # (C) total cost = distance + penalty
            cost_matrix = dist_matrix + penalty_matrix

            # (D) new assignment = argmin cost
            new_assignments = np.argmin(cost_matrix, axis=1)

            # Check if we changed
            if np.all(new_assignments == assignments):
                # Converged
                assignments = new_assignments
                if verbose:
                    print(f"[PCKmeans] Converged at iteration {iteration+1}")
                break

            assignments = new_assignments

            # -----------------------------------------------------------------
            # (E) Update cluster centers
            # -----------------------------------------------------------------
            new_centers = np.zeros((self.k, dim), dtype=np.float32)
            counts = np.zeros(self.k, dtype=int)

            for s_idx in range(supernode_count):
                c = assignments[s_idx]
                new_centers[c] += supernode_centroids[s_idx]
                counts[c] += 1

            # Re-init any empty cluster to a random supernode
            for c in range(self.k):
                if counts[c] == 0:
                    new_centers[c] = supernode_centroids[rng.integers(supernode_count)]
                else:
                    new_centers[c] /= counts[c]

            cluster_centers = new_centers

        # ---------------------------------------------------------------------
        # 4) Expand final supernode assignments to per-sample labels
        # ---------------------------------------------------------------------
        final_labels = np.zeros(n_samples, dtype=int)
        for s_idx, c in enumerate(assignments):
            for idx in supernode_indices[s_idx]:
                final_labels[idx] = c

        self.labels_ = final_labels
        self.cluster_centers_ = cluster_centers

        # Build images_lists (list of lists of sample indices per cluster)
        images_lists = [[] for _ in range(self.k)]
        for i, c in enumerate(final_labels):
            images_lists[c].append(i)
        self.images_lists = images_lists

        # ---------------------------------------------------------------------
        # 5) Optional: check constraint violations
        # ---------------------------------------------------------------------
        must_link_violations, cannot_link_violations = self._compute_constraint_violations(final_labels)
        
        if verbose:
            print("\nConstraint Violation Details:")
            print(f"Must-link violations ({len(must_link_violations)}/{len(self.constraints.get('must_link', []))})")
            for i, j in must_link_violations[:5]:  # Show first 5 violations
                print(f"  Pair ({i},{j}): assigned to clusters {final_labels[i]}, {final_labels[j]}")
                
            print(f"\nCannot-link violations ({len(cannot_link_violations)}/{len(self.constraints.get('cannot_link', []))})")
            for i, j in cannot_link_violations[:5]:  # Show first 5 violations
                print(f"  Pair ({i},{j}): both assigned to cluster {final_labels[i]}")

        # After convergence, verify must-link satisfaction
        if verbose:
            must_link_violations = []
            for (i, j) in must_link:
                if final_labels[i] != final_labels[j]:
                    must_link_violations.append((i, j))
            print(f"\nMust-link constraint satisfaction: "
                  f"{len(must_link) - len(must_link_violations)}/{len(must_link)} "
                  f"({100 * (1 - len(must_link_violations)/len(must_link)):.1f}%)")

        # ---------------------------------------------------------------------
        # 6) Compute final pseudo-loss (sum of squared distances)
        # ---------------------------------------------------------------------
        total_loss = 0.0
        for i in range(n_samples):
            c = final_labels[i]
            diff = x_data[i] - cluster_centers[c]
            total_loss += 0.5 * np.dot(diff, diff)

        # ---------------------------------------------------------------------
        # 7) Optional plotting
        # ---------------------------------------------------------------------
        if self.plot and (fig is not None) and (axes is not None) and (true_labels is not None) and (epoch is not None):
            self._plot_clusters(fig, axes, x_data, final_labels, true_labels, epoch)

        elapsed = time.time() - start_time
        if verbose:
            print(f"[PCKmeans] Finished in {elapsed:.2f}s, final_loss={total_loss:.4f}, #supernodes={supernode_count}")

        return total_loss

    def _plot_clusters(self, fig, axes, features, cluster_labels, true_labels, epoch):
        """
        A TSNE-based plot for cluster assignments vs. true labels.
        Highlights labeled samples and constraint violations within the plotted subset.
        """
        # force inclusion of labeled samples
        labeled_array = np.array(list(self.labeled_indices))
        labeled_array = labeled_array[labeled_array < features.shape[0]]  # clamp any out-of-range
        n_samples = features.shape[0]
        fraction = 0.05
        n_subset = int(fraction * n_samples)
        rng = np.random.default_rng()

        # pick the remaining random subset from unlabeled
        unlabeled_array = np.setdiff1d(np.arange(n_samples), labeled_array)
        n_random = max(0, n_subset - len(labeled_array))
        random_samples = rng.choice(unlabeled_array, size=n_random, replace=False) if n_random else np.array([], dtype=int)
        subset_idx = np.concatenate([labeled_array, random_samples])
        subset_idx_set = set(subset_idx)

        X_2d = TSNE(n_components=2, random_state=42).fit_transform(features[subset_idx])
        sub_assign = cluster_labels[subset_idx]
        sub_true = true_labels[subset_idx]

        # create mapping from original idx to subset idx
        orig_to_subset = {orig: sub for sub, orig in enumerate(subset_idx)}

        # Clear old axes
        for ax in axes:
            ax.clear()

        # Plot cluster assignments
        unique_clusters = np.unique(sub_assign)
        for cl_id in unique_clusters:
            pts = X_2d[sub_assign == cl_id]
            axes[0].scatter(pts[:, 0], pts[:, 1], label=f"C {cl_id}", alpha=0.6)
        axes[0].set_title(f"PCKmeans Clusters (Epoch {epoch})")
        axes[0].legend()

        # Plot true labels
        unique_gt = np.unique(sub_true)
        for label_id in unique_gt:
            pts = X_2d[sub_true == label_id]
            axes[1].scatter(pts[:, 0], pts[:, 1], label=f"Label {label_id}", alpha=0.6)
        axes[1].set_title(f"True Labels (Epoch {epoch})")
        axes[1].legend()

        # Highlight labeled samples
        for idx in self.labeled_indices:
            if idx in subset_idx_set:
                sub_idx = orig_to_subset[idx]
                axes[0].scatter(X_2d[sub_idx, 0], X_2d[sub_idx, 1], c='black', s=100, alpha=0.3, label='_nolegend_')

        # Draw constraint violations
        must_link = self.constraints.get('must_link', [])
        cannot_link = self.constraints.get('cannot_link', [])

        # Must-link violations
        for (i, j) in must_link:
            if i in subset_idx_set and j in subset_idx_set:  # only if both points are in our subset
                if cluster_labels[i] != cluster_labels[j]:  # violation
                    sub_i, sub_j = orig_to_subset[i], orig_to_subset[j]
                    axes[0].plot([X_2d[sub_i, 0], X_2d[sub_j, 0]], 
                               [X_2d[sub_i, 1], X_2d[sub_j, 1]], 
                               'r--', linewidth=1, label='_nolegend_')
                    # Highlight the points involved
                    axes[0].scatter([X_2d[sub_i, 0], X_2d[sub_j, 0]], 
                                  [X_2d[sub_i, 1], X_2d[sub_j, 1]], 
                                  c='red', s=100, alpha=0.3, label='_nolegend_')

        # Cannot-link violations
        for (i, j) in cannot_link:
            if i in subset_idx_set and j in subset_idx_set:  # only if both points are in our subset
                if cluster_labels[i] == cluster_labels[j]:  # violation
                    sub_i, sub_j = orig_to_subset[i], orig_to_subset[j]
                    axes[0].plot([X_2d[sub_i, 0], X_2d[sub_j, 0]], 
                               [X_2d[sub_i, 1], X_2d[sub_j, 1]], 
                               'b--', linewidth=1, label='_nolegend_')
                    # Highlight the points involved
                    axes[0].scatter([X_2d[sub_i, 0], X_2d[sub_j, 0]], 
                                  [X_2d[sub_i, 1], X_2d[sub_j, 1]], 
                                  c='blue', s=100, alpha=0.3, label='_nolegend_')

        # Add legend for constraint violations
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='r', linestyle='--', label='Must-link violation'),
            Line2D([0], [0], color='b', linestyle='--', label='Cannot-link violation'),
            Line2D([0], [0], marker='o', color='w', label='Labeled sample',
                   markerfacecolor='none', markeredgecolor='g', markersize=10)
        ]
        axes[0].legend(handles=legend_elements, loc='upper right')

        fig.tight_layout()
        plt.pause(2)





