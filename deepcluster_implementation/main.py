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
from torchvision.datasets import MNIST, EMNIST
import clustering
from util import AverageMeter, Logger, UnifLabelSampler, create_sparse_labels, create_constraints
import models
import os
import random
import matplotlib.pyplot as plt
import json
import tqdm

def get_device(args):
    '''
    Get the device to use for training
    '''
    if args['device'] == 'cuda' and torch.cuda.is_available():
        print("Using CUDA")
        return torch.device("cuda")
    elif args['device'] == 'mps' and torch.backends.mps.is_available():
        print("Using MPS (Metal Performance Shaders)")
        return torch.device("mps")
    else:
        print("Using CPU")
        return torch.device("cpu")   

def get_dataset(args):
    '''
    Get the dataset to use for training
    '''
    # Select dataset based on args
    if args.get('dataset', 'MNIST').upper() == 'EMNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1736,), (0.3248,))  # MNIST/EMNIST mean and std
        ])
        dataset = EMNIST(
            root=args['data'],
            split='letters',  # Default split; adjust as needed
            train=True,
            download=True,
            transform=transform
        )
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST/EMNIST mean and std
        ])
        dataset = MNIST(root=args['data'], train=True, download=True, transform=transform)

    if args['verbose']:
        print(f"Dataset: {args.get('dataset', 'MNIST').upper()}")
        print("Sample shape: " + str(dataset[0][0].shape))
    
    return dataset


def get_model(args, device):
    '''
    Get the model to use for training
    '''
    # Create Model
    if args['verbose']:
        print('Architecture:', args['arch'])

    model = models.__dict__[args['arch']]()
    
    fd = int(model.top_layer.weight.size()[1])
    model.top_layer = None
    # Move the model to the correct device
    model = model.to(device)
    
    # Wrap the model's feature extractor in DataParallel
    model.features = torch.nn.DataParallel(model.features)

    # Ensure the entire model is on the correct device
    model = model.to(device)

    return model, fd

def save_params(args):
    '''
    Save the parameters to a json file
    '''
    with open(os.path.join(args['exp'], 'params.json'), 'w') as f:
        json.dump(args, f, indent=4)

def calculate_constraint_weights(n, fraction, nmb_cluster, violation_weight):
    '''
    Calculate the number of constraints to use based on the fraction of the dataset
    '''
    if fraction == 0:
        return 0
    w = 1./(n*fraction/nmb_cluster) # normalize by number of clusters
    w *= violation_weight # multiply by violation weight
    return w


def run_experiment(args):
    # Fix random seeds
    torch.manual_seed(args['seed'])
    torch.cuda.manual_seed_all(args['seed'])
    np.random.seed(args['seed'])

    # Save the parameters to a json file
    save_params(args)
    # Create a unique folder for each run
    run_folder = args['exp']  # Use the folder passed from run_experiments directly
    os.makedirs(os.path.join(run_folder, 'visualizations'), exist_ok=True)

    metrics_log = {
        'nmi_true': [],
        'ari_true': [],
        'nmi_prev': [],
        'silhouette': [],
        'dbi': []
    }

    device = get_device(args)

    dataset = get_dataset(args)

    # Create constraints for PCKmeans if label fraction is constant over all epochs
    w = 1.
    if isinstance(args['label_fraction'], float): 
        w = calculate_constraint_weights(len(dataset), args['label_fraction'], args['nmb_cluster'], args['violation_weight'])
        partial_labeled_data, labeled_indices = create_sparse_labels(args, args['label_fraction'], dataset)
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
                                                          max_iter=args['pckmeans_iters'],
                                                          w=w,
                                                          device=device, 
                                                          plot=args['plot_clusters'])

    # Logging setup
    cluster_log = Logger(os.path.join(run_folder, 'clusters'))

    # if plot_clusters, create figure
    if args['plot_clusters']:
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # Start Training
    for epoch in range(args['epochs']):
        start_epoch_time = time.time()
        # Remove head
        model.top_layer = None
        model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])

        # Compute features
        features = compute_features(train_loader, model, args)

        # Extract true labels
        true_labels = np.array([label for _, label in dataset])

        # If dynamic label fraction, create constraints
        if isinstance(args['label_fraction'], list):
            deepcluster.w = calculate_constraint_weights(len(dataset), args['label_fraction'][epoch], args['nmb_cluster'], args['violation_weight'])
            partial_labeled_data, labeled_indices = create_sparse_labels(args, args['label_fraction'][epoch], dataset)
            constraints = create_constraints(args, partial_labeled_data, labeled_indices)

        # Cluster features and visualize
        if args['verbose']: print('Clustering features')
        save_path = os.path.join(run_folder, 'visualizations', f"epoch_{epoch}.pdf")
        clustering_loss = deepcluster.cluster(fig, axes, features, 
                                              true_labels=true_labels, 
                                              epoch=epoch, 
                                              constraints=constraints,
                                              labeled_indices=labeled_indices,
                                              verbose=args['verbose'], 
                                              save_path=save_path)

        # Assign pseudo-labels
        if args['verbose']: print('Assigning pseudo labels')
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
        loss = train(train_dataloader, model, criterion, optimizer, epoch, args)

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

def compute_features(dataloader, model, args):
    """Extract features from the dataset using PyTorch for GPU acceleration."""
    device = args.get('device', 'cpu')  # Default to CPU if not specified
    model.eval()

    batch_time = AverageMeter()  # Initialize the AverageMeter for batch timing
    features = []  # Use a list to collect features

    for input_tensor, _ in tqdm.tqdm(dataloader, desc="Compute Features", disable=not args['verbose']):
        batch_start = time.time()

        # Move input tensor to the correct device
        input_var = input_tensor.to(device)

        # Forward pass
        with torch.no_grad():
            aux = model(input_var).detach()  # Detach to avoid keeping gradients

        # Collect the features
        features.append(aux)

        # Update batch timing
        batch_time.update(time.time() - batch_start)

    # Concatenate all features into a single tensor
    features = torch.cat(features, dim=0)

    # Return the features as a CPU tensor for further processing if necessary
    return features.cpu().numpy()




def train(loader, model, criterion, optimizer, epoch, args):
    """Train the CNN with multiple passes over the training set."""
    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()
    
    device = args.get('device', 'cpu')  # Default to CPU if not specified

    model.train()

    # Optimizer for the last fully connected layer
    optimizer_tl = torch.optim.SGD(
        model.top_layer.parameters(),
        lr=args['lr'],
        weight_decay=10 ** args['wd'],
    )

    end = time.time()

    progress_bar = tqdm.tqdm(loader, desc=f"Epoch {epoch}", disable=not args['verbose'])

    for i, (input_tensor, target) in enumerate(progress_bar):
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

        # Update tqdm progress bar with additional metrics
        progress_bar.set_postfix({
            'Loss': f'{losses.avg:.4f}',
            'Accuracy': f'{accuracy:.4f}',
            'Batch Time': f'{batch_time.avg:.3f}s'
        })

    return losses.avg



if __name__ == "__main__":
    default_args = {
        'data': './data',  # Path to dataset
        'dataset': 'MNIST',  # Dataset to use
        'arch': 'simplecnn',  # Model architecture
        'clustering': 'PCKmeans',
        # 'clustering': 'Kmeans',
        'nmb_cluster': 10,  # Number of clusters (10 for MNIST digits)
        'lr': 5e-2,
        'wd': -5,
        'reassign': 3.0,
        'workers': 2,
        'epochs': 6,
        'batch': 256,
        'momentum': 0.9,
        'resume': '',  # Path to checkpoint
        'checkpoints': 25000,
        'seed': 31,
        'exp': './experiment',
        'verbose': True,
        'device': 'cpu',  # Set to 'cuda', 'mps', or 'cpu'
        'plot_clusters' : True,
        'label_fraction': 0.001,  # Fraction of the dataset to use for testing, float if constant fraction, or list of length=epochs if label fraction dynamically changes during training
        # 'label_fraction': [0, 0, 0, 0.001, 0.001, 0.002],
        'cannot_link_fraction': 0.1,  # This is the fraction you want to use (1.0 = all constraints)
        'must_link_fraction': 1.0, # This is the fraction you want to use (1.0 = all constraints)
        'label_pattern': 'random',
        'nmb_labeled_clusters': 5, # Number of clusters to use for labeled data
        'label_noise': 0.0,
        'pckmeans_iters': 5,
        'granularity': 1, # Granularity-sized label cluster
        'custom_clusters': None,
        'violation_weight': 2.,
    }
    run_experiment(default_args)