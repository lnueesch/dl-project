# DL 2024 Course Project

Marek Landert  
Johannes Nüesch  
Lukas Nüesch  

# Semi-supervised Clustering with DeepCluster

## Motivation
Datasets often contain sparse or missing labels, impacting clustering performance. This project extends **DeepCluster** with **PCKMeans**, enabling the use of sparse labels through soft constraints.

## Model
We enhance DeepCluster by modifying the CNN loss function and integrating PCKMeans for label-guided clustering.

## Datasets
- **MNIST**: Benchmark dataset for image clustering.

## Baseline
Fully unsupervised **DeepCluster** with k-means serves as the baseline.

## Experiments
### Label Properties
- **Sparsity**: Varying percentages of labeled data (1%, 5%, 10%, 20%).
- **Patterns**: Cluster-wise or feature-conditioned label sparsity.
- **Noise**: Robustness to mislabeled data.
- **Granularity**: Coarse-grained vs. fine-grained labels.
- **Dynamic Labels**: Simulating temporal label availability.

### Training Strategies
- **Pretraining**: Random vs. pretrained weights.
- **Loss Variants**: Testing alternative loss functions.
- **Dynamic Constraints**: Adjusting constraints during training.

## Evaluation Metrics
- **Clustering**: ARI, NMI, and Cluster Purity.
- **Latent Space**: Visualizations using t-SNE or UMAP.
- **Robustness**: Sensitivity to noise, sparsity, and outliers.

## Expected Outcomes
1. Insights into label sparsity effects on clustering.
2. Recommendations for semi-supervised clustering with sparse labels.
