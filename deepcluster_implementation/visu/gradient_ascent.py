# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import os
from scipy.ndimage import gaussian_filter
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

sys.path.insert(0, '..')
from util import load_model

# parser = argparse.ArgumentParser(description='Gradient ascent visualisation')
# parser.add_argument('--model', type=str, help='Model')
# parser.add_argument('--arch', type=str, default='alexnet', choices=['alexnet', 'vgg16'], help='arch')
# parser.add_argument('--conv', type=int, default=1, help='convolutional layer')
# parser.add_argument('--exp', type=str, default='', help='path to res')
# parser.add_argument('--lr', type=float, default=3, help='learning rate (default: 3)')
# parser.add_argument('--wd', type=float, default=0.00001, help='weight decay (default: 10^-5)')
# parser.add_argument('--sig', type=float, default=0.3, help='gaussian blur (default: 0.3)')
# parser.add_argument('--step', type=int, default=5, help='number of iter between gaussian blurs (default: 5)')
# parser.add_argument('--niter', type=int, default=1000, help='total number of iterations (default: 1000)')
# parser.add_argument('--idim', type=int, default=224, help='size of input image (default: 224)')



CONV = {'alexnet': [96, 256, 384, 384, 256],
        'vgg16':     [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512],
        'simplecnn': [32, 64]}


def main():
    args = {
        'model': '../experiment/checkpoint.pth.tar',
        'arch': 'simplecnn',
        'conv': 2,
        'exp': 'results',
        'lr': 30,
        'wd': 0.00001,
        'sig': 0.3,
        'step': 5,
        'niter': 1000,
        'idim': 28
    }

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    # sanity check
    if args['arch'] == 'alexnet':
        assert args['conv'] < 6
    elif args['arch'] == 'vgg16':
        assert args['conv'] < 14
    elif args['arch'] == 'simplecnn':
        assert args['conv'] < 3

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

    def gradient_ascent(f):
        print(f)
        sys.stdout.flush()
        fname_out = '{0}/layer{1}-channel{2}.jpeg'.format(repo, args['conv'], f)

        img_noise = np.random.normal(size=(args['idim'], args['idim'], 1)) * 20 + 128
        img_noise = img_noise.astype('float32')
        inp = transforms.ToTensor()(img_noise)
        inp = torch.unsqueeze(inp, 0)

        for it in range(args['niter']):
            x = torch.autograd.Variable(inp.to(device), requires_grad=True)
            out = forward(model, args['conv']-1, f, x)
            criterion = nn.CrossEntropyLoss()
            filt_var = torch.autograd.Variable(torch.ones(1).long()*f).to(device)
            output = out.mean(3).mean(2)
            loss = - criterion(output, filt_var) - args['wd']*torch.norm(x)**2

            # compute gradient
            loss.backward()

            # normalize gradient
            grads = x.grad.data.cpu()
            grads = grads.div(torch.norm(grads)+1e-8)

            # apply gradient
            inp = inp.add(args['lr']*grads)
            #print shape of inp
            #print(inp.squeeze(0).shape)

            # gaussian blur
            if it % args['step'] == 0:
                inp = gaussian_filter(inp.squeeze(0).numpy().transpose((2, 1, 0)),
                                       sigma=(args['sig'], args['sig'], 0))
                inp = torch.unsqueeze(torch.from_numpy(inp).float().transpose(2, 0), 0)

            #print(inp.shape)
            # save image at the last iteration
            if it == args['niter'] - 1:
                a = deprocess_image(inp.numpy())[0]
                Image.fromarray(a).save(fname_out)
                print('Saved layer{0}-channel{1}'.format(args['conv'], f))

    for f in range(CONV[args['arch']][args['conv'] - 1]):
        gradient_ascent(f)


def deprocess_image(x):
    x = x[0, :, :, :]
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to greyscale array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def forward(model, layer, channel, x):
    if model.sobel is not None:
        x = model.sobel(x)
    count = 0
    for y, m in enumerate(model.features.modules()):
        if not isinstance(m, nn.Sequential):
            x = m(x)
            if isinstance(m, nn.Conv2d):
                if count == layer:
                    res = x
            if isinstance(m, nn.ReLU):
                if count == layer:
                    # check if channel is not activated
                    if x[:, channel, :, :].mean().data.cpu().numpy() == 0:
                        return res
                    return x
                count = count + 1


if __name__ == '__main__':
    main()
