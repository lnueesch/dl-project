# DL 2024 course project

Marek Landert  
Johannes Nüesch  
Lukas Nüesch  

# Semi-supervised Non-parametric Clustering with DeepCluster

## Motivation
In many real world scenarios we are presented with datasets that only contain sparse labels or even miss labels for certain classes. 
This project aims to extend the **DeepCluster** architecture, which is fully Unsupervised is its original form, to work with sparse label data and an unknown number
clusters. 

To summarize we will:
- Add support for partial labels
- Add support for unknown number of clusters

Using the additional label information we aim to find the number of clusters jointly with the cluster assignments.

## Model
We aim to base our approaches on **DeepCluster** and extend and extend it to our problem. For label and unknown number of cluster support
 we will experiment with different architectures.

## Datasets
Since we want to solve the probelems jointly we wont be able to use pretrained networks. Therefore smaller and simpler datasets like **MNIST** are more
appropriate.

## Baseline
Our Baseline will be the fully unsupervised DeepCluster without any labels but with known number of clusters.


## Experiments
We aim to show that we can achieve comparable performance even with unknown number of clusters. Further we aim to improve accuracy using labelled data.


