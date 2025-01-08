# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from util import AverageMeter, learning_rate_decay, load_model, Logger

# Define parameters directly
args = {
    'data': './data',  # Path to dataset
    'model': 'experiment/checkpoint.pth.tar',  # Path to model
    'conv': 1,  # Convolutional layer to train logistic regression on
    'tencrops': False,
    'lr': 0.01,
    'batch_size': 256,
    'epochs': 50,
    'workers': 4,
    'verbose': True,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

class LogisticRegression(nn.Module):
    def __init__(self, conv, num_labels):
        super(LogisticRegression, self).__init__()
        if conv == 1:
            self.av_pool = nn.AvgPool2d(4, stride=4, padding=0)
            s = 288
        elif conv == 2:
            self.av_pool = nn.AvgPool2d(3, stride=3, padding=0)
            s = 9216
        elif conv == 3:
            self.av_pool = nn.AvgPool2d(2, stride=2, padding=0)
            s = 9216
        elif conv == 4:
            self.av_pool = nn.AvgPool2d(2, stride=2, padding=0)
            s = 9600
        elif conv == 5:
            self.av_pool = nn.AvgPool2d(2, stride=2, padding=0)
            s = 9216
        self.linear = nn.Linear(s, num_labels)

    def forward(self, x):
        x = self.av_pool(x)
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        return self.linear(x)

def forward(x, model, conv):
    if hasattr(model, 'sobel') and model.sobel is not None:
        x = model.sobel(x)
    count = 1
    for m in model.features.modules():
        if not isinstance(m, nn.Sequential):
            x = m(x)
        if isinstance(m, nn.ReLU):
            if count == conv:
                return x
            count = count + 1
    return x

def main():
    # Load dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(args['data'], train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST(args['data'], train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True, num_workers=args['workers'])
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=args['workers'])

    # Load model
    model = load_model(args['model'])
    model = model.to(args['device'])

    # Define logistic regression layer
    reglog = LogisticRegression(args['conv'], 10)
    reglog = reglog.to(args['device'])

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss().to(args['device'])
    optimizer = torch.optim.SGD(reglog.parameters(), args['lr'], momentum=0.9, weight_decay=1e-4)

    # Training and validation
    for epoch in range(args['epochs']):
        train(train_loader, model, reglog, criterion, optimizer, epoch)
        validate(val_loader, model, reglog, criterion)

def train(train_loader, model, reglog, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        learning_rate_decay(optimizer, len(train_loader) * epoch + i, args['lr'])

        input = input.to(args['device'])
        target = target.to(args['device'])

        output = forward(input, model, args['conv'])
        output = reglog(output)
        loss = criterion(output, target)

        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if args['verbose'] and i % 100 == 0:
            print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                  f'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  f'Prec@5 {top5.val:.3f} ({top5.avg:.3f})')

def validate(val_loader, model, reglog, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()
    softmax = nn.Softmax(dim=1).to(args['device'])
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input = input.to(args['device'])
        target = target.to(args['device'])

        with torch.no_grad():
            output = forward(input, model, args['conv'])
            output = reglog(output)
            output = softmax(output)

        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))
        loss = criterion(output, target)
        losses.update(loss.item(), input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if args['verbose'] and i % 100 == 0:
            print(f'Validation: [{i}/{len(val_loader)}]\t'
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                  f'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  f'Prec@5 {top5.val:.3f} ({top5.avg:.3f})')

    return top1.avg, top5.avg, losses.avg

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == "__main__":
    main()
