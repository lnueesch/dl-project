import time
import numpy as np
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

# Define parameters directly
args = {
    'data': './data',  # Path to dataset
    'arch': 'simplecnn',  # Model architecture
    'sobel': False,
    'clustering': 'Kmeans',
    'nmb_cluster': 10,  # Number of clusters (10 for MNIST digits)
    'lr': 0.05,
    'wd': -5,
    'reassign': 1.0,
    'workers': 4,
    'epochs': 50,
    'batch': 256,
    'momentum': 0.9,
    'resume': '',  # Path to checkpoint
    'checkpoints': 25000,
    'seed': 31,
    'exp': './experiment',
    'verbose': True,
    'device': 'mps',  # Set to 'cuda', 'mps', or 'cpu'
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
    fraction = 0.1  # Use 10% of the dataset

    # Load MNIST dataset
    full_dataset = MNIST(root=args['data'], train=True, download=True, transform=transform)

    # Create a subset of the dataset
    subset_size = int(len(full_dataset) * fraction)
    indices = random.sample(range(len(full_dataset)), subset_size)
    dataset = Subset(full_dataset, indices)

    # DataLoader
    train_loader = DataLoader(
        dataset,
        batch_size=args['batch'],
        num_workers=args['workers'],
        shuffle=True,
        pin_memory=True
    )

    # Create Model
    if args['verbose']:
        print('Architecture:', args['arch'])
    model = models.__dict__[args['arch']](sobel=args['sobel'])
    model.top_layer = None
    model.features = torch.nn.DataParallel(model.features)
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


    # Clustering
    deepcluster = clustering.__dict__[args['clustering']](args['nmb_cluster'])

    # Logging setup
    cluster_log = Logger(os.path.join(args['exp'], 'clusters'))

    # Start Training
    for epoch in range(args['epochs']):
        # Compute features
        features = compute_features(train_loader, model, len(dataset), device)

        # Cluster features
        clustering_loss = deepcluster.cluster(features, verbose=args['verbose'])

        # Assign pseudo-labels
        train_dataset = clustering.cluster_assign(deepcluster.images_lists, dataset.data)

        # Uniformly sample targets
        sampler = UnifLabelSampler(int(args['reassign'] * len(train_dataset)), deepcluster.images_lists)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args['batch'],
            num_workers=args['workers'],
            sampler=sampler,
            pin_memory=True
        )

        # Train network with pseudo-labels
        loss = train(train_dataloader, model, criterion, optimizer, epoch, device)

        # Log and save progress
        cluster_log.log(deepcluster.images_lists)
        save_checkpoint(epoch, model, optimizer, args)


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


def compute_features(dataloader, model, N, device):
    """Extract features from the dataset."""
    if args['verbose']:
        print('Compute features')
    batch_time = AverageMeter()
    end = time.time()
    model.eval()

    for i, (input_tensor, _) in enumerate(dataloader):
        input_var = torch.autograd.Variable(input_tensor.to(device), volatile=True)
        aux = model(input_var).data.cpu().numpy()

        if i == 0:
            features = np.zeros((N, aux.shape[1]), dtype='float32')

        aux = aux.astype('float32')
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
    """Train the CNN."""
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

    end = time.time()
    for i, (input_tensor, target) in enumerate(loader):
        data_time.update(time.time() - end)

        # Transfer data to GPU
        target = target.to(device)
        input_var = torch.autograd.Variable(input_tensor.to(device))
        target_var = torch.autograd.Variable(target)

        # Forward pass
        output = model(input_var)
        loss = criterion(output, target_var)

        # Record loss
        losses.update(loss.item(), input_tensor.size(0))

        # Backward pass and optimization
        optimizer.zero_grad()
        optimizer_tl.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer_tl.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args['verbose'] and (i % 200) == 0:
            print(f"Epoch: [{epoch}][{i}/{len(loader)}]\t"
                  f"Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                  f"Loss: {losses.val:.4f} ({losses.avg:.4f})")

    return losses.avg


if __name__ == "__main__":
    main(args)