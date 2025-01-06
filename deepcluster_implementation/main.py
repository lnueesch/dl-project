import time
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
# from active_semi_clustering.active.pairwise_constraints import ExampleOracle, ExploreConsolidate, MinMax
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import MNIST
import clustering
from util import AverageMeter, Logger, UnifLabelSampler
import models
import os
import random
import matplotlib.pyplot as plt
from collections import defaultdict

# Define parameters directly
args = {
    'data': './data',  # Path to dataset
    'arch': 'simplecnn',  # Model architecture
    'sobel': False,
    'clustering': 'PCKmeans',
    'nmb_cluster': 10,  # Number of clusters (10 for MNIST digits)
    'lr': 5e-1,
    'wd': -5,
    'reassign': 10.0,
    'workers': 2,
    'epochs': 5,
    'batch': 256,
    'momentum': 0.9,
    'resume': '',  # Path to checkpoint
    'checkpoints': 25000,
    'seed': 31,
    'exp': './experiment',
    'verbose': True,
    'device': 'cuda',  # Set to 'cuda', 'mps', or 'cpu'
    'plot_clusters' : True,
}

def main(args):
    # Fix random seeds
    torch.manual_seed(args['seed'])
    torch.cuda.manual_seed_all(args['seed'])
    np.random.seed(args['seed'])

    # Set device based on user input
    if args['device'] == 'cuda' and torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    elif args['device'] == 'mps' and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Data Preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])

    # Fraction of the dataset to use for testing
    fraction = 0.2  # Use 10% of the dataset

    # Load MNIST dataset
    dataset = MNIST(root=args['data'], train=True, download=True, transform=transform)
    # dataset = Subset(dataset, np.arange(0, len(dataset), 1))
    dataset = Subset(dataset, random.sample(range(len(dataset)), int(fraction * len(dataset))))

    # print shape of a single sample
    print("sample shape: " + str(dataset[0][0].shape))

    # DataLoader
    train_loader = DataLoader(
        dataset,
        batch_size=args['batch'],
        num_workers=args['workers'],
        shuffle=False,
        pin_memory=True
    )

    # Create Model
    if args['verbose']:
        print('Architecture:', args['arch'])
    model = models.__dict__[args['arch']](sobel=args['sobel'])

    fd = int(model.top_layer.weight.size()[1])
    model.top_layer = None
    # Move the model to the correct device
    model = model.to(device)

    # Wrap the model's feature extractor in DataParallel
    model.features = torch.nn.DataParallel(model.features)

    # Ensure the entire model is on the correct device
    model = model.to(device)
    cudnn.benchmark = True

    # Optimizer
    optimizer = torch.optim.SGD(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=args['lr'],
        momentum=args['momentum'],
        weight_decay=10 ** args['wd'],
    )

    # Loss Function
    criterion = nn.CrossEntropyLoss().to(device)

    # Generate Pairwise Constrains
    sparse_dataset = create_sparse_labels(dataset, fraction=0.01, pattern="random", noise=0.0, seed=42)
    pairwise_constraints = generate_constraints(sparse_dataset)
    print("created pairwise constraints:", len(pairwise_constraints[0]), len(pairwise_constraints[1]))

    # Save 2 images from pairwise must-link constraints
    # save_image(dataset, pairwise_constraints[0][0][0], "image0.png")
    # save_image(dataset, pairwise_constraints[0][0][1], "image1.png")

    # Clustering
    deepcluster = clustering.__dict__[args['clustering']](args['nmb_cluster'], device, plot=args['plot_clusters'], pairwise_constraints=pairwise_constraints)

    # Logging setup
    cluster_log = Logger(os.path.join(args['exp'], 'clusters'))

    # if plot_clusters, create figure
    if args['plot_clusters']:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Start Training
    print("Start training")
    for epoch in range(args['epochs']):
        # Remove head
        model.top_layer = None
        model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])

        # Compute features
        features = compute_features(train_loader, model, len(dataset), device)

        # Extract true labels
        true_labels = np.array([label for _, label in dataset])

        # Cluster features and visualize
        if args['verbose']:
            print('Clustering features')
        save_path = os.path.join(args['exp'], 'visualizations', f"epoch_{epoch}.png")
        clustering_loss = deepcluster.cluster(fig, axes, features, true_labels=true_labels, epoch=epoch, verbose=args['verbose'])

        # Assign pseudo-labels
        if args['verbose']:
            print('Assigning pseudo labels')
        train_dataset = clustering.cluster_assign(deepcluster.images_lists, dataset.dataset)


        # Uniformly sample targets
        sampler = UnifLabelSampler(int(args['reassign'] * len(train_dataset)), deepcluster.images_lists)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args['batch'],
            num_workers=args['workers'],
            sampler=sampler,
            pin_memory=True
        )

        # set last fully connected layer
        mlp = list(model.classifier.children())
        mlp.append(nn.ReLU(inplace=True).to(device))
        model.classifier = nn.Sequential(*mlp)
        model.top_layer = nn.Linear(fd, len(deepcluster.images_lists))
        model.top_layer.weight.data.normal_(0, 0.01)
        model.top_layer.bias.data.zero_()
        model.top_layer.to(device)

        # Train network with pseudo-labels
        end = time.time()
        loss = train(train_dataloader, model, criterion, optimizer, epoch, device)

        # print log
        if args['verbose']:
            print('###### Epoch [{0}] ###### \n'
                  'Time: {1:.3f} s\n'
                  'Clustering loss: {2:.3f} \n'
                  'ConvNet loss: {3:.3f}'
                  .format(epoch, time.time() - end, clustering_loss, loss))
            try:
                nmi = normalized_mutual_info_score(
                    clustering.arrange_clustering(deepcluster.images_lists),
                    clustering.arrange_clustering(cluster_log.data[-1])
                )
                print('NMI against previous assignment: {0:.3f}'.format(nmi))
            except IndexError:
                pass
            print('####################### \n')
        # save running checkpoint
        torch.save({'epoch': epoch + 1,
                    'arch': args['arch'],
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict()},
                   os.path.join(args['exp'], 'checkpoint.pth.tar'))

        # save cluster assignments
        cluster_log.log(deepcluster.images_lists)


def generate_constraints(sparse_dataset):
    """
    Generate must-link and cannot-link constraints from a partially labeled dataset.
    
    Args:
        dataset: a list of (image, label) tuples where label == -1 means unlabeled.
    
    Returns:
        constraints: a tuple containing:
            - must_link: a list of pairs (i, j) where data[i] and data[j] must belong to the same cluster.
            - cannot_link: a list of pairs (i, j) where data[i] and data[j] must belong to different clusters.
    """
    # oracle = ExampleOracle(true_labels, max_queries_cnt=10)
    # active_learner = MinMax(n_clusters=10)
    # active_learner.fit(xb, oracle=oracle)
    # pairwise_constraints = active_learner.pairwise_constraints_
    # return pairwise_constraints
    
    # Group indices by label, ignoring unlabeled data (-1)
    labeled_data = defaultdict(list)
    for idx, (_, label) in enumerate(sparse_dataset):
        if label != -1:
            labeled_data[label].append(idx)
    
    # Generate must-link constraints
    must_link = []
    for indices in labeled_data.values():
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                must_link.append((indices[i], indices[j]))
    
    # Generate cannot-link constraints
    cannot_link = []
    labels = list(labeled_data.keys())
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            for idx1 in labeled_data[labels[i]]:
                for idx2 in labeled_data[labels[j]]:
                    cannot_link.append((idx1, idx2))
    
    return must_link, cannot_link


def create_sparse_labels(dataset, fraction=0.1, pattern="random", noise=0.0, seed=42):
    """
    Create a partially labeled dataset from a fully labeled dataset.
    
    Args:
        dataset: a PyTorch Dataset where dataset[i] = (image, label)
        fraction (float): fraction of labels we want to *keep* (e.g., 0.1 means 10%).
        pattern (str): "random", "missing_classes", or custom logic
        noise (float): fraction of kept labels that we corrupt (0.0 means no noise)
        seed (int): random seed for reproducibility

    Returns:
        new_dataset: a list or custom dataset of (image, label) where label==-1 means unlabeled
    """
    rng = random.Random(seed)
    total_size = len(dataset)

    # Convert dataset to list for easy manipulation
    data_list = [(dataset[i][0], dataset[i][1]) for i in range(total_size)]
    
    if pattern == "random":
        # random subset of labeled indices
        n_labeled = int(fraction * total_size)
        labeled_indices = rng.sample(range(total_size), n_labeled)
        labeled_indices = set(labeled_indices)
    elif pattern == "missing_classes":
        # e.g. remove all labels for half the classes
        # This is just an example; adapt as needed.
        # Let's find unique classes:
        all_labels = [label for (_, label) in data_list]
        classes = list(set(all_labels))
        rng.shuffle(classes)
        half = len(classes) // 2
        unlabeled_classes = set(classes[:half])
        
        labeled_indices = []
        for i, (img, lbl) in enumerate(data_list):
            if lbl not in unlabeled_classes:
                labeled_indices.append(i)
        labeled_indices = set(labeled_indices)
        
        # You could also keep fraction within those classes, etc.
    else:
        # fallback or custom pattern
        labeled_indices = set(range(total_size))  # by default label all
    
    # Build new dataset with partial labels
    new_dataset = []
    for i, (img, lbl) in enumerate(data_list):
        if i not in labeled_indices:
            # remove label
            new_dataset.append((img, -1))
        else:
            # keep label
            new_dataset.append((img, lbl))
    
    # Introduce noise in the kept labels (optional)
    if noise > 0.0:
        # the portion of labeled samples we will corrupt
        # e.g., noise=0.1 means 10% of the labeled samples get assigned random label
        labeled_inds_list = list(labeled_indices)
        rng.shuffle(labeled_inds_list)
        n_noisy = int(noise * len(labeled_inds_list))
        noisy_part = labeled_inds_list[:n_noisy]
        all_possible_labels = list(set([l for (x, l) in data_list]))
        for idx in noisy_part:
            # pick a random label from all_possible_labels (could exclude the true one if desired)
            corrupted_lbl = rng.choice(all_possible_labels)
            new_dataset[idx] = (new_dataset[idx][0], corrupted_lbl)

    return new_dataset


def save_image(dataset, index, filepath):
    """
    Save an image from the dataset to a file.
    
    Args:
        dataset: a list of (image, label) tuples.
        index (int): index of the image to save.
        filepath (str): path to save the image file (e.g., "image.png").
    """
    # Get the image and label
    image, label = dataset[index]
    
    # Convert PyTorch tensors or similar objects to NumPy arrays
    if hasattr(image, 'numpy'):
        image = image.numpy()
    
    # Handle channels-first format (e.g., shape (1, 28, 28) for grayscale or (3, H, W) for RGB)
    if len(image.shape) == 3 and image.shape[0] in [1, 3]:  
        image = image.squeeze(0) if image.shape[0] == 1 else image.transpose(1, 2, 0)
    
    # Save the image
    plt.imshow(image, cmap='gray' if image.ndim == 2 else None)
    plt.title(f"Label: {label}")
    plt.axis('off')  # Hide axes for better visualization
    plt.savefig(filepath, bbox_inches='tight', pad_inches=0)
    plt.close()  # Close the figure to free memory


def save_checkpoint(epoch, model, optimizer, args):
    """Save a checkpoint of the model."""
    checkpoint_path = os.path.join(args['exp'], f'checkpoint_epoch_{epoch}.pth.tar')
    torch.save({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, checkpoint_path)
    if args['verbose']:
        print(f"Checkpoint saved at {checkpoint_path}")

# def compute_features(dataloader, model, N, device):
#     """Return raw images from the dataset."""
#     if args['verbose']:
#         print('Returning raw images as features for testing')

#     # Initialize features as an array to store raw images
#     features = None

#     for i, (input_tensor, _) in enumerate(dataloader):
#         # Flatten the input images (e.g., from [B, 1, 28, 28] to [B, 784])
#         input_flat = input_tensor.view(input_tensor.size(0), -1).cpu().numpy()

#         if features is None:
#             features = np.zeros((N, input_flat.shape[1]), dtype='float32')

#         if i < len(dataloader) - 1:
#             features[i * args['batch']: (i + 1) * args['batch']] = input_flat
#         else:
#             features[i * args['batch']:] = input_flat

#     return features

def compute_features(dataloader, model, N, device):
    """Extract features from the dataset."""
    if args['verbose']:
        print('Compute features')
    batch_time = AverageMeter()
    end = time.time()
    model.eval()

    features = None

    for i, (input_tensor, _) in enumerate(dataloader):
        # Move input tensor to the correct device
        input_var = input_tensor.to(device)

        # Forward pass
        with torch.no_grad():
            aux = model(input_var).data.cpu().numpy()

        if features is None:
            features = np.zeros((N, aux.shape[1]), dtype='float32')

        aux = aux.astype('float32')
        if np.any(np.isnan(aux)) or np.any(np.isinf(aux)):
            raise ValueError("NaN or Inf detected in computed features")

        if i < len(dataloader) - 1:
            features[i * args['batch']: (i + 1) * args['batch']] = aux
        else:
            features[i * args['batch']:] = aux

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args['verbose'] and (i % 200) == 0:
            print(f"{i}/{len(dataloader)}\tTime: {batch_time.val:.3f} ({batch_time.avg:.3f})")

    

    return features




def train(loader, model, criterion, optimizer, epoch, device):
    """Train the CNN with multiple passes over the training set."""
    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()

    model.train()

    # Optimizer for the last fully connected layer
    optimizer_tl = torch.optim.SGD(
        model.top_layer.parameters(),
        lr=args['lr'],
        weight_decay=10 ** args['wd'],
    )
    # optimizer_tl = torch.optim.Adam(
    #     model.top_layer.parameters(),
    #     lr=args['lr']
    # )

    end = time.time()
    for i, (input_tensor, target) in enumerate(loader):
        data_time.update(time.time() - end)

        # Move inputs and targets to the correct device
        input_var = input_tensor.to(device)
        target_var = target.to(device)

        # Forward pass
        output = model(input_var)
        loss = criterion(output, target_var)

        # Measure accuracy on pseudo-labels
        _, predicted = torch.max(output, 1)
        correct = (predicted == target_var).sum().item()
        accuracy = correct / input_tensor.size(0)

        # Record loss
        losses.update(loss.item(), input_tensor.size(0))

        # Backward pass and optimization
        optimizer.zero_grad()
        optimizer_tl.zero_grad()
        loss.backward()
        # optimizer.step()
        optimizer_tl.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args['verbose'] and (i % 100) == 0:
            print(' Epoch: [{0}][{1}/{2}]\t'
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Accuracy: {accuracy:.4f}'
                    .format(epoch, i, len(loader),
                            batch_time=batch_time, data_time=data_time,
                            loss=losses, accuracy=accuracy))

    return losses.avg



if __name__ == "__main__":
    main(args)
