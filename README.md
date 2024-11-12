# DL 2024 course project

Marek  
Johannes Nüesch  
Lukas Nüesch  

# Clustering with Varying Label Quality and Quantity

## Objective
This project investigates how the quality and quantity of labels impact the performance of deep learning clustering algorithms. We assess clustering effectiveness across different levels of label availability, label granularity, and label accuracy.

## Model
We use **Semi-Supervised Contrastive Clustering (SSCC)** as our primary model. SSCC is well-suited for this project because:
- It leverages **transfer learning** with pre-trained models (e.g., ResNet or EfficientNet), significantly reducing training time.
- It is effective for **semi-supervised clustering**, allowing us to incorporate partial labels efficiently.

## Datasets
- **MNIST**: A simpler dataset for initial experimentation with digit clustering.
- **ImageNet Subset**: Used to explore clustering performance on a larger, more complex dataset with hierarchical labels.

## Variables
1. **Percentage of Available Labels**: Experiment with different percentages of labeled data (e.g., 10%, 30%, 50%, 100%).
2. **Granularity of Labels**: Test different levels of label specificity (e.g., broader superclasses vs. fine-grained classes).
3. **Percentage of Incorrect Labels** (optional): Introduce noise by randomly assigning incorrect labels to a portion of the data.

## Baseline
- **Fully Unsupervised Clustering**: Run SSCC without any labels to establish a baseline for comparison.

## Evaluation Metrics
1. **Normalized Mutual Information (NMI)**: Measures mutual dependence between clusters and true labels.
2. **Adjusted Rand Index (ARI)**: Evaluates clustering accuracy compared to true labels.
3. **Silhouette Score**: Assesses cluster cohesion and separation.
4. **Cluster Purity**: Evaluates how well clusters align with ground truth classes.

## Visualization
Use **t-SNE** or **UMAP** to visualize cluster distributions and observe the impact of varying label quality and quantity on cluster formation.

## Project Structure
1. **Data Preprocessing**: Prepare MNIST and ImageNet subsets, create high- and low-granularity labels, and add label noise if applicable.
2. **Model Training**: Fine-tune SSCC using different label configurations and apply transfer learning for efficiency.
3. **Evaluation and Visualization**: Calculate metrics and visualize clusters to analyze how label variations affect clustering quality.


