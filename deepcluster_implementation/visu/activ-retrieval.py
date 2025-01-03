import argparse
import os
from shutil import copyfile
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

sys.path.insert(0, '..')
from util import load_model

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# def parse_args():
    # parser = argparse.ArgumentParser(description='Retrieve images with maximal activations')
    # parser.add_argument('--data', type=str, help='path to dataset')
    # parser.add_argument('--model', type=str, help='Model')
    # parser.add_argument('--conv', type=int, default=1, help='convolutional layer')
    # parser.add_argument('--exp', type=str, default='', help='path to res')
    # parser.add_argument('--count', type=int, default=9, help='save this many images')
    # parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
    # return parser.parse_args()


def forward(model, my_layer, x):
    if model.sobel is not None:
        x = model.sobel(x)
    layer = 1
    res = {}
    for m in model.features.modules():
        if not isinstance(m, nn.Sequential):
            x = m(x)
            if isinstance(m, nn.ReLU):
                if layer == my_layer:
                    for channel in range(int(x.size()[1])):
                        key = 'layer' + str(layer) + '-channel' + str(channel)
                        res[key] = torch.squeeze(x.mean(3).mean(2))[:, channel]
                    return res
                layer = layer + 1
    return res

def main(args):
    # create repo
    repo = os.path.join(args['exp'], 'conv' + str(args['conv']))
    if not os.path.isdir(repo):
        os.makedirs(repo)

    # build model
    model = load_model(args['model'])
    model.to(device)
    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    # load data
    normalize = transforms.Normalize(mean=[0.1307], std=[0.3081])
    tra = [transforms.Resize(28),
           transforms.ToTensor(),
           normalize]

    # dataset
    dataset = datasets.MNIST(args['data'], train=False, transform=transforms.Compose(tra), download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, num_workers=args['workers'])

    # keys are filters and value are arrays with activation scores for the whole dataset
    layers_activations = {}
    for i, (input_tensor, _) in enumerate(dataloader):
        input_var = torch.autograd.Variable(input_tensor.to(device), volatile=True)
        activations = forward(model, args['conv'], input_var)

        if i == 0:
            layers_activations = {filt: np.zeros(len(dataset)) for filt in activations}
        if i < len(dataloader) - 1:
            e_idx = (i + 1) * 256
        else:
            e_idx = len(dataset)
        s_idx = i * 256
        for filt in activations:
            layers_activations[filt][s_idx: e_idx] = activations[filt].cpu().data.numpy()

        if i % 100 == 0:
            print('{0}/{1}'.format(i, len(dataloader)))

    # save top N images for each filter
    for filt in layers_activations:
        repofilter = os.path.join(repo, filt)
        if not os.path.isdir(repofilter):
            os.mkdir(repofilter)
        top = np.argsort(layers_activations[filt])[::-1]
        if args['count'] > 0:
            top = top[:args['count']]

        for pos, img in enumerate(top):
            src, _ = dataset.data[img]
            img = Image.fromarray(src.numpy(), mode='L')
            img.save(os.path.join(repofilter, "{}_{}.png".format(pos, img)))

if __name__ == '__main__':
    args = {
        'data': './data',  # Path to dataset
        'model': './simplecnn',  # Path to model
        'conv': 1,  # Convolutional layer
        'exp': './experiment',  # Path to res
        'count': 9,  # Save this many images
        'workers': 4  # Number of data loading workers
    }
    main(args)