# DL 2024 Course Project

Marek Landert  
Johannes Nüesch  
Lukas Nüesch  

# Exploring Dataset Label Properties in Semi-Supervised Clustering

This work extends the DeepCluster framework
by incorporating sparse label information through
PCK-means must-link and cannot-link constraints.
Using the MNIST dataset, we analyze how label
sparsity, distribution, noise, granularity, and dy-
namic availability affect clustering performance.
Our results show that even minimal supervision
(1% labeled data or less) significantly enhances
clustering performance, even under moderate
noise. However, balanced class labels are crucial
for optimal performance, whereas coarse labels
offer only marginal gains. These findings high-
light the potential of semi-supervised clustering.

# How to run experiments
All experiments that are conducted can be reproduced by running.

python run_experiments.py

