import time
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
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

# Define parameters directly
args = {
    'data': './data',  # Path to dataset
    'arch': 'simplecnn',  # Model architecture
    'sobel': False,
    'clustering': 'PCKmeans',
    # 'clustering': 'Kmeans',
    'nmb_cluster': 10,  # Number of clusters (10 for MNIST digits)
    'lr': 5e-2,
    'wd': -5,
    'reassign': 3.0,
    'workers': 4,
    'epochs': 10,
    'batch': 256,
    'momentum': 0.9,
    'resume': '',  # Path to checkpoint
    'checkpoints': 25000,
    'seed': 31,
    'exp': './experiment',
    'verbose': True,
    'device': 'mps',  # Set to 'cuda', 'mps', or 'cpu'
    'plot_clusters' : True,
    'label_fraction': 0.001,  # Fraction of the dataset to use for testing
    'cannot_link_fraction': 0.1  # This is the fraction you want to use (1.0 = all constraints)
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
    # fraction = 0.7  # Use 10% of the dataset

    # Load MNIST dataset
    dataset = MNIST(root=args['data'], train=True, download=True, transform=transform)
    
    # PROBLEMATIC: This could cause index misalignment
    # dataset = Subset(dataset, np.arange(0, len(dataset), 1))
    
    # Create partially labeled dataset BEFORE any subsetting
    if args['verbose']:
        print('Creating partially labeled dataset')
    partial_labeled_data, labeled_indices = create_sparse_labels(
        dataset,
        fraction=args['label_fraction'], 
        pattern="random",
        noise=0.0,
        seed=2024
    )

    if args['verbose']:
        print('Creating constraints')
    constraints = create_constraints(
        dataset,
        labeled_indices,
        cl_fraction=args['cannot_link_fraction'],
        ml_fraction=1,
        seed=2024
    )

    print("Number of ml constraints:", len(constraints[0]))
    print("Number of cl constraints:", len(constraints[1]))

    # print shape of a single sample
    print("sample shape: " + str(dataset[0][0].shape))

    # DataLoader
    train_loader = DataLoader(
        dataset,
        # partial_labeled_data,
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

    # Clustering
    deepcluster = clustering.__dict__[args['clustering']](k=args['nmb_cluster'], 
                                                          device=device, 
                                                          plot=args['plot_clusters'], 
                                                          constraints=constraints,
                                                          labeled_indices=labeled_indices)

    # Create a unique folder for each run
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_folder = os.path.join(args['exp'], f"run_{timestamp}")
    os.makedirs(os.path.join(run_folder, 'visualizations'), exist_ok=True)

    # Save parameters
    with open(os.path.join(run_folder, 'params.json'), 'w') as f:
        json.dump(args, f, indent=4)

    # Logging setup
    cluster_log = Logger(os.path.join(run_folder, 'clusters'))

    # if plot_clusters, create figure
    if args['plot_clusters']:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Start Training
    for epoch in range(args['epochs']):
        start_epoch_time = time.time()
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
        save_path = os.path.join(run_folder, 'visualizations', f"epoch_{epoch}.png")
        clustering_loss = deepcluster.cluster(fig, axes, features, true_labels=true_labels, epoch=epoch, verbose=args['verbose'])

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
        loss = train(train_dataloader, model, criterion, optimizer, epoch, device)

        # print log
        if args['verbose']:
            print('###### Epoch [{0}] ###### \n'
                  'Time: {1:.3f} s\n'
                  'Clustering loss: {2:.3f} \n'
                  'ConvNet loss: {3:.3f}'
                  .format(epoch, time.time() - start_epoch_time, clustering_loss, loss))
            try:
                nmi = normalized_mutual_info_score(
                    clustering.arrange_clustering(deepcluster.images_lists),
                    clustering.arrange_clustering(cluster_log.data[-1])
                )
                print('NMI against previous assignment: {0:.3f}'.format(nmi))
            except IndexError:
                pass
            nmi_true = normalized_mutual_info_score(
                clustering.arrange_clustering(deepcluster.images_lists),
                true_labels
            )
            print('NMI against true labels: {0:.3f}'.format(nmi_true))
            print('####################### \n')
        # save running checkpoint
        torch.save({'epoch': epoch + 1,
                    'arch': args['arch'],
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict()},
                   os.path.join(run_folder, f'checkpoint_epoch_{epoch}.pth.tar'))

        # save cluster assignments
        cluster_log.log(deepcluster.images_lists)

    # save final model
    final_checkpoint_path = os.path.join(run_folder, 'final_model.pth.tar')
    torch.save({
        'epoch': args['epochs'],
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, final_checkpoint_path)

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

        if args['verbose'] and (i % 64) == 0:
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
    main(args)