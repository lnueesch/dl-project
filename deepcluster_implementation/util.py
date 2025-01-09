# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import pickle

import numpy as np
import torch
from torch.utils.data.sampler import Sampler

import models
import random


def load_model(path):
    """Loads model and return it without DataParallel table."""
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path)

        # size of the top layer
        N = checkpoint['state_dict']['top_layer.bias'].size()

        # build skeleton of the model
        sob = 'sobel.0.weight' in checkpoint['state_dict'].keys()
        model = models.__dict__[checkpoint['arch']](sobel=sob, out=int(N[0]))

        # deal with a dataparallel table
        def rename_key(key):
            if not 'module' in key:
                return key
            return ''.join(key.split('.module'))

        checkpoint['state_dict'] = {rename_key(key): val
                                    for key, val
                                    in checkpoint['state_dict'].items()}

        # load weights
        model.load_state_dict(checkpoint['state_dict'])
        print("Loaded")
    else:
        model = None
        print("=> no checkpoint found at '{}'".format(path))
    return model



def create_sparse_labels(args, dataset):
    '''
    Create a partially labeled dataset from a fully labeled dataset.
    '''
    if args['verbose']:
        print('Creating partially labeled dataset')
    rng = random.Random(args['seed'])
    total_size = len(dataset)
    
    # Create clean labeled dataset first
    labeled_indices = set(rng.sample(range(total_size), int(args['label_fraction'] * total_size)))
    
    # Initialize all labels as -1 (unlabeled)
    new_labels = [-1] * total_size
    
    # First assign correct labels
    for idx in labeled_indices:
        _, label = dataset[idx]
        new_labels[idx] = label
    
    # Then apply noise to a subset if requested
    noise = args['label_noise']
    if noise > 0.0:
        noisy_count = int(len(labeled_indices) * noise)
        noisy_indices = rng.sample(list(labeled_indices), noisy_count)
        
        all_labels = list(range(10))  # MNIST has 10 classes
        for idx in noisy_indices:
            true_label = new_labels[idx]
            possible_wrong = [l for l in all_labels if l != true_label]
            new_labels[idx] = rng.choice(possible_wrong)
            print(f"Noise applied: idx={idx}, true={true_label} -> noisy={new_labels[idx]}")
    
    # Create new dataset with these labels
    new_dataset = [(dataset[i][0], new_labels[i]) for i in range(total_size)]

    #print how many labeled samples by counting non -1 labels 
    labeled_count = sum(1 for label in new_labels if label != -1)
    print(f"Total labeled samples: {labeled_count} out of {total_size} ({labeled_count/total_size*100:.2f}%)")
    
    return new_dataset, labeled_indices

import random
from collections import defaultdict

import random
from collections import defaultdict
from itertools import combinations

def create_constraints(args, dataset, labeled_indices):
    """
    Create must-link and cannot-link constraints from a semi-supervised dataset.

    Parameters
    ----------
    dataset : Sequence
        A dataset where each item is (image, label). Labels are accessed via dataset[idx][1].
    labeled_indices : list of int
        The indices for which the labels are known (the semi-supervised subset).
    cl_fraction : float, optional
        Fraction of all possible cannot-link constraints to include (default=1.0).
    seed : int, optional
        Seed for random number generator, for reproducibility.

    Returns
    -------
    must_links : list of tuple(int, int)
        List of index pairs (i, j) that must link (i.e., have the same label).
    cannot_links : list of tuple(int, int)
        List of index pairs (i, j) that cannot link (i.e., have different labels),
        sampled at the specified fraction of all possible pairs.
    """

    if args['verbose']:
        print('Creating constraints')
    
    random.seed(args['seed'])

    cl_fraction = args['cannot_link_fraction']
    ml_fraction = args['must_link_fraction']

    # 1. Group labeled indices by their label.
    label_dict = defaultdict(list)
    for idx in labeled_indices:
        # label is the second element in dataset[idx]
        lbl = dataset[idx][1]
        label_dict[lbl].append(idx)

    # 2. Construct all must-link constraints.
    #    For each label, create pairwise combinations among indices having that label.
    must_links = []
    for lbl, idxs in label_dict.items():
        # Only need combinations if there are at least 2 indices for this label
        if len(idxs) > 1:
            must_links.extend(combinations(idxs, 2))

    # 3. Construct cannot-link constraints across labels.
    #    We'll sample them so that we only take 'cl_fraction' of all possible cross-label pairs.
    cannot_links = []
    all_labels = sorted(label_dict.keys())
    for i in range(len(all_labels)):
        for j in range(i + 1, len(all_labels)):
            lbl_i = all_labels[i]
            lbl_j = all_labels[j]

            idxs_i = label_dict[lbl_i]
            idxs_j = label_dict[lbl_j]

            # Possible cross-label pairs = cartesian product of idxs_i and idxs_j
            n1, n2 = len(idxs_i), len(idxs_j)
            total_pairs = n1 * n2
            if total_pairs == 0:
                continue

            # Number of cannot-link pairs to sample from this cross-label set
            n_to_sample = int(cl_fraction * total_pairs)
            # Edge case: if fraction is 0 or the rounding yields 0, skip
            if n_to_sample <= 0:
                continue

            if n_to_sample >= total_pairs:
                # Take all pairs (lbl_i vs lbl_j)
                for id_i in idxs_i:
                    for id_j in idxs_j:
                        cannot_links.append((id_i, id_j))
            else:
                # Randomly sample 'n_to_sample' unique pairs without generating all first.
                #   1) We represent each pair by an integer from 0..(total_pairs-1).
                #   2) Then decode that integer back to (i1, i2).
                chosen_indices = random.sample(range(total_pairs), n_to_sample)
                for c in chosen_indices:
                    i1 = c // n2
                    i2 = c % n2
                    cannot_links.append((idxs_i[i1], idxs_j[i2]))

    if args['verbose']:
        print("Number of ml constraints:", len(must_links))
        print("Number of cl constraints:", len(cannot_links))
        
    return must_links, cannot_links





class UnifLabelSampler(Sampler):
    """Samples elements uniformely accross pseudolabels.
        Args:
            N (int): size of returned iterator.
            images_lists: dict of key (target), value (list of data with this target)
    """

    def __init__(self, N, images_lists):
        self.N = N
        self.images_lists = images_lists
        self.indexes = self.generate_indexes_epoch()

    def generate_indexes_epoch(self):
        nmb_non_empty_clusters = 0
        for i in range(len(self.images_lists)):
            if len(self.images_lists[i]) != 0:
                nmb_non_empty_clusters += 1

        size_per_pseudolabel = int(self.N / nmb_non_empty_clusters) + 1
        res = np.array([])

        for i in range(len(self.images_lists)):
            # skip empty clusters
            if len(self.images_lists[i]) == 0:
                continue
            indexes = np.random.choice(
                self.images_lists[i],
                size_per_pseudolabel,
                replace=(len(self.images_lists[i]) <= size_per_pseudolabel)
            )
            res = np.concatenate((res, indexes))

        np.random.shuffle(res)
        res = list(res.astype('int'))
        if len(res) >= self.N:
            return res[:self.N]
        res += res[: (self.N - len(res))]
        return res

    def __iter__(self):
        return iter(self.indexes)

    def __len__(self):
        return len(self.indexes)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def learning_rate_decay(optimizer, t, lr_0):
    for param_group in optimizer.param_groups:
        lr = lr_0 / np.sqrt(1 + lr_0 * param_group['weight_decay'] * t)
        param_group['lr'] = lr


class Logger(object):
    """ Class to update every epoch to keep trace of the results
    Methods:
        - log() log and save
    """

    def __init__(self, path):
        self.path = path
        self.data = []

    def log(self, train_point):
        self.data.append(train_point)
        with open(os.path.join(self.path), 'wb') as fp:
            pickle.dump(self.data, fp, -1)
