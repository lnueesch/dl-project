# DL 2024 Course Project

Marek Landert  
Johannes Nüesch  
Lukas Nüesch  

# Semi-supervised Clustering with DeepCluster

## Motivation
In many real-world scenarios, datasets often contain sparse labels or lack labels for certain classes entirely. 
Understanding how different types of label sparsity influence clustering performance is crucial for designing robust semi-supervised learning systems. 

This project aims to investigate the impact of sparse labels on clustering performance by building on the **DeepCluster** architecture. 
While DeepCluster operates fully unsupervised using k-means, we will extend it by replacing k-means with **cop-kmeans**, which allows the incorporation of sparse labels during training.

Through a series of experiments, we aim to understand how dataset properties, label characteristics, and training strategies influence the clustering outcomes. 
These insights could guide future approaches for semi-supervised and constrained clustering.

## Model
Our model is based on **DeepCluster**, extended with **cop-kmeans** to leverage sparse labels for enhanced performance.

## Datasets
We will conduct our experiments on the following datasets:
- **MNIST**: A benchmark dataset for image clustering with clear cluster boundaries.
- **CIFAR-10**: A more complex dataset with higher variability and challenging cluster separation.
- **Synthetic Datasets**: Controlled datasets (e.g., Gaussian Mixture Models, Swiss Roll) to study clustering in a structured and interpretable manner.

## Baseline
Our baseline will be the original fully unsupervised **DeepCluster** architecture using k-means without any label support. This will serve as a reference point to evaluate the improvements introduced by sparse labels and cop-kmeans.

## Experiments
We will explore various label properties and training strategies to systematically evaluate their effect on clustering performance:

### Label Properties
1. **Overall Sparsity**:
   - Varying the percentage of labeled images in the dataset (e.g., 1%, 5%, 10%, 20%).
2. **Sparsity Patterns**:
   - **Cluster-wise Sparsity**: Labels are available for only certain clusters.
   - **Feature-conditioned Sparsity**: Labels are provided based on feature distributions (e.g., high-density or high-variance regions).
3. **Number of Outliers**:
   - Assessing robustness to unlabeled or misclassified outlier data points.
4. **Noise**:
   - Introducing incorrect labels to evaluate the model's ability to handle mislabeled data.
5. **Label Granularity**:
   - Testing the effect of varying label granularity (e.g., coarse-grained vs. fine-grained class labels).

### Training Strategies
1. **Pretraining**:
   - Comparing performance with randomly initialized weights vs. pretraining on similar datasets.
2. **Loss Variants**:
   - Experimenting with different loss functions to incorporate label support more effectively (e.g., contrastive losses or pairwise constraints).
3. **Dynamic Data**:
   - Introducing labels dynamically during training to simulate real-world scenarios where labeled data becomes available over time (temporal sparsity).

## Evaluation Metrics
To evaluate our model's performance, we will use:
- **Clustering Metrics**:
  - Adjusted Rand Index (ARI),
  - Normalized Mutual Information (NMI),
  - Cluster Purity.
- **Latent Space Analysis**:
  - Visualizing the latent space using techniques like t-SNE or UMAP to understand cluster separation.
- **Robustness**:
  - Sensitivity analysis for noise, outliers, and sparsity patterns.

## Expected Outcomes
1. Insights into the impact of various types of label sparsity on clustering performance.
2. Recommendations for handling sparse labels in semi-supervised clustering tasks.

