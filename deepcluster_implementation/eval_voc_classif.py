# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from sklearn import metrics
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from util import AverageMeter, load_model
from eval_linear import accuracy

# Define parameters directly
args = {
    'data': './data',  # Path to MNIST dataset
    'model': 'experiment/checkpoint.pth.tar',  # Model to evaluate
    'eval_random_crops': 1,  # If true, eval on 10 random crops, otherwise eval on 10 fixed crops
    'lr': 0.003,  # Learning rate
    'wd': 1e-6,  # Weight decay
    'batch_size': 256,
    'workers': 4,
    'verbose': True,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'nit': 5,  # Number of training iterations
    'stepsize': 5000,  # Decay step
    'min_scale': 0.1,  # Scale
    'max_scale': 0.5,  # Scale
    'fc6_8': 1,  # If true, train only the final classifier
}

def main():
    print(args)

    # Set random seed
    torch.manual_seed(31)
    if args['device'] == 'cuda':
        torch.cuda.manual_seed_all(31)

    # Load model
    model = load_model(args['model'])
    model = model.to(args['device'])

    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load dataset
    train_dataset = datasets.MNIST(args['data'], train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST(args['data'], train=False, transform=transform)

    # only use 1/10 of the data
    train_dataset = torch.utils.data.Subset(train_dataset, np.arange(0, len(train_dataset), 10))
    val_dataset = torch.utils.data.Subset(val_dataset, np.arange(0, len(val_dataset), 10))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True, num_workers=args['workers'])
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=args['workers'])

    optimizer = torch.optim.SGD(model.parameters(), lr=args['lr'], weight_decay=args['wd'])

    print('Start training')
    for it in range(args['nit']):
        print('Iteration', it)
        model = train(train_loader, model, optimizer, args)

    print('Evaluation')
    if args['eval_random_crops']:
        transform_eval = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(28, scale=(args['min_scale'], args['max_scale']), ratio=(1, 1)), 
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        transform_eval = transforms.Compose([
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    val_dataset = datasets.MNIST(args['data'], train=False, transform=transform_eval)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=args['workers'])

    evaluate(val_loader, model, args['eval_random_crops'])

def train(loader, model, optimizer, args):
    model.train()
    criterion = nn.CrossEntropyLoss().to(args['device'])
    
    for i, (input, target) in enumerate(loader):
        input = input.to(args['device'])
        target = target.to(args['device'])
        
        optimizer.zero_grad()
        output = model(input)
        
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    return model

def evaluate(loader, model, eval_random_crops):
    model.eval()
    gts = []
    scr = None  # Initialize scr dynamically on the first pass

    for crop in range(9 * eval_random_crops + 1):
        for i, (input, target) in enumerate(loader):
            input = input.to(args['device'])

            # Forward pass without gradient computation
            with torch.no_grad():
                output = model(input).cpu().numpy()

            # Initialize scr on the first pass
            if scr is None:
                scr = np.zeros((len(loader.dataset), output.shape[1]), dtype=np.float32)

            # Assign indices for this batch
            start_idx = i * loader.batch_size
            end_idx = start_idx + output.shape[0]

            # Aggregate scores
            scr[start_idx:end_idx] += output

            if crop == 0:
                gts.extend(target.numpy())

    gts = np.array(gts)
    aps = []

    # Compute average precision for each class
    for i in range(10):  # MNIST has 10 classes
        ap = metrics.average_precision_score((gts == i).astype(int), scr[:, i])
        aps.append(ap)

    # Print mean and individual APs
    print(np.mean(aps), '  ', ' '.join(['%0.2f' % a for a in aps]))

if __name__ == "__main__":
    main()

