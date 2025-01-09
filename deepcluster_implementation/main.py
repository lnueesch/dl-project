import time
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics import silhouette_score, davies_bouldin_score
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import MNIST
import clustering
from util import AverageMeter, Logger, UnifLabelSampler, create_sparse_labels, create_constraints
import models
import os
import random
import matplotlib.pyplot as plt
import json

def get_device(args):
    '''
    Get the device to use for training
    '''
    if args['device'] == 'cuda' and torch.cuda.is_available():
        print("Using CUDA")
        return torch.device("cuda")
    elif args['device'] == 'mps' and torch.backends.mps.is_available():
        print("Using MPS (Metal Performance Shaders)")
        device = torch.device("mps")
    else:
        print("Using CPU")
        device = torch.device("cpu")   

def get_dataset(args):
    '''
    Get the dataset to use for training
    '''

    # Data Preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])

    # Load MNIST dataset
    dataset = MNIST(root=args['data'], train=True, download=True, transform=transform)

    if args['verbose']:
        print("sample shape: " + str(dataset[0][0].shape))
    
    return dataset

def get_model(args, device):
    '''
    Get the model to use for training
    '''
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

    return model, fd


def run_experiment(args):
    # Fix random seeds
    torch.manual_seed(args['seed'])
    torch.cuda.manual_seed_all(args['seed'])
    np.random.seed(args['seed'])

    device = get_device(args)

    dataset = get_dataset(args)

    partial_labeled_data, labeled_indices = create_sparse_labels(args, dataset)

    constraints = create_constraints(args, partial_labeled_data, labeled_indices)

    train_loader = DataLoader(
        dataset,
        # partial_labeled_data,
        batch_size=args['batch'],
        num_workers=args['workers'],
        shuffle=False,
        pin_memory=True
    )

    model, fd = get_model(args, device)
    
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
    deepcluster = clustering.__dict__[args['clustering']](k=args['nmb_cluster'], 
                                                          max_iter=args['kmeans_iters'],
                                                          device=device, 
                                                          plot=args['plot_clusters'], 
                                                          constraints=constraints,
                                                          labeled_indices=labeled_indices)

    # Create a unique folder for each run
    run_folder = args['exp']  # Use the folder passed from run_experiments directly
    os.makedirs(os.path.join(run_folder, 'visualizations'), exist_ok=True)

    # Save parameters
    with open(os.path.join(run_folder, 'params.json'), 'w') as f:
        json.dump(args, f, indent=4)

    # Logging setup
    cluster_log = Logger(os.path.join(run_folder, 'clusters'))

    # if plot_clusters, create figure
    if args['plot_clusters']:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    metrics_log = {
        'nmi_true': [],
        'ari_true': [],
        'nmi_prev': [],
        'silhouette': [],
        'dbi': []
    }

    # Start Training
    for epoch in range(args['epochs']):
        start_epoch_time = time.time()
        # Remove head
        model.top_layer = None
        model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])

        # Compute features
        features = compute_features(train_loader, model, len(dataset), device, args)

        # Extract true labels
        true_labels = np.array([label for _, label in dataset])

        # Cluster features and visualize
        if args['verbose']:
            print('Clustering features')
        save_path = os.path.join(run_folder, 'visualizations', f"epoch_{epoch}.png")
        clustering_loss = deepcluster.cluster(fig, axes, features, true_labels=true_labels, epoch=epoch, verbose=args['verbose'], save_path=save_path)

        # Assign pseudo-labels
        if args['verbose']:
            print('Assigning pseudo labels')
        train_dataset = clustering.cluster_assign(deepcluster.images_lists, dataset)


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
        loss = train(train_dataloader, model, criterion, optimizer, epoch, device, args)

        # print log
        if args['verbose']:
            print('###### Epoch [{0}] ###### \n'
                  'Time: {1:.3f} s\n'
                  'Clustering loss: {2:.3f} \n'
                  'ConvNet loss: {3:.3f}'
                  .format(epoch, time.time() - start_epoch_time, clustering_loss, loss))
            
        
        # extract current labels
        labels_cur = clustering.arrange_clustering(deepcluster.images_lists)

        try:
            labels_prev = clustering.arrange_clustering(cluster_log.data[-1])
            nmi_prev = normalized_mutual_info_score(labels_cur, labels_prev)
        except IndexError:
            nmi_prev = None

        # compute metrics vs ground truth
        nmi_true = normalized_mutual_info_score(labels_cur, true_labels)
        ari_true = adjusted_rand_score(true_labels, labels_cur)
        silhouette = silhouette_score(features, labels_cur) if len(np.unique(labels_cur)) > 1 else None
        dbi = davies_bouldin_score(features, labels_cur) if len(np.unique(labels_cur)) > 1 else None

        # Cast any np.float32 to Python’s built-in float before appending to the metrics_log
        nmi_true = float(nmi_true)
        ari_true = float(ari_true)
        nmi_prev = float(nmi_prev) if nmi_prev is not None else None
        silhouette = float(silhouette) if silhouette is not None else None
        dbi = float(dbi) if dbi is not None else None

        if args['verbose']:
            print(f"nmi_true: {nmi_true:.3f}")
            print(f"ari_true: {ari_true:.3f}")
            print(f"nmi_prev: {nmi_prev:.3f}" if nmi_prev is not None else "nmi_prev: None")
            print(f"silhouette: {silhouette:.3f}" if silhouette is not None else "silhouette: None")
            print(f"dbi: {dbi:.3f}" if dbi is not None else "dbi: None")
            print('####################### \n')

        # store metrics
        metrics_log['nmi_true'].append(nmi_true)
        metrics_log['ari_true'].append(ari_true)
        metrics_log['nmi_prev'].append(nmi_prev)
        metrics_log['silhouette'].append(silhouette)
        metrics_log['dbi'].append(dbi)

        # save running checkpoint
        checkpoint_dir = os.path.join(run_folder, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save({'epoch': epoch + 1,
                'arch': args['arch'],
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()},
               os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth.tar'))

        # save cluster assignments
        cluster_log.log(deepcluster.images_lists)

    # save final model
    final_checkpoint_path = os.path.join(run_folder, 'final_model.pth.tar')
    torch.save({
        'epoch': args['epochs'],
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, final_checkpoint_path)

    # save metrics
    with open(os.path.join(run_folder, 'metrics.json'), 'w') as f:
        json.dump(metrics_log, f, indent=4)

    #close all open figures
    plt.close('all')

    print(f"Results saved at: {run_folder}")


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

def compute_features(dataloader, model, N, device, args):
    """Extract features from the dataset."""
    if args['verbose']:
        print('Compute features')
    batch_time = AverageMeter()
    end = time.time()
    model.eval()

    features = None

    # Timing the initial setup
    setup_start = time.time()
    for i, (input_tensor, _) in enumerate(dataloader):
        if i == 0:
            setup_end = time.time()
            print(f"Initial setup time: {setup_end - setup_start:.3f} seconds")

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

        if args['verbose'] and ((i % 64) == 0 or i == len(dataloader) - 1):
            print(f"{i}/{len(dataloader)}\tTime: {batch_time.val:.3f} ({batch_time.avg:.3f})")

    return features




def train(loader, model, criterion, optimizer, epoch, device, args):
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
        optimizer.step()
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
    default_args = {
        'data': './data',  # Path to dataset
        'arch': 'simplecnn',  # Model architecture
        'sobel': False,
        'clustering': 'PCKmeans',
        # 'clustering': 'Kmeans',
        'nmb_cluster': 10,  # Number of clusters (10 for MNIST digits)
        'lr': 5e-2,
        'wd': -5,
        'reassign': 3.0,
        'workers': 12,
        'epochs': 10,
        'batch': 256,
        'momentum': 0.9,
        'resume': '',  # Path to checkpoint
        'checkpoints': 25000,
        'seed': 31,
        'exp': './experiment',
        'verbose': True,
        'device': 'cpu',  # Set to 'cuda', 'mps', or 'cpu'
        'plot_clusters' : True,
        'label_fraction': 0.001,  # Fraction of the dataset to use for testing
        'cannot_link_fraction': 0.1,  # This is the fraction you want to use (1.0 = all constraints)
        'must_link_fraction': 1.0,  # This is the fraction you want to use (1.0 = all constraints)
        'label_pattern': 'random',
        'label_noise': 0.0,
        'kmeans_iters': 10
    }
    run_experiment(default_args)