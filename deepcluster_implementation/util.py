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



def create_sparse_labels(dataset, fraction=0.01, pattern="random", noise=0.0, seed=42):
    rng = random.Random(seed)
    total_size = len(dataset)
    
    # Create clean labeled dataset first
    labeled_indices = set(rng.sample(range(total_size), int(fraction * total_size)))
    
    # Initialize all labels as -1 (unlabeled)
    new_labels = [-1] * total_size
    
    # First assign correct labels
    for idx in labeled_indices:
        _, label = dataset[idx]
        new_labels[idx] = label
    
    # Then apply noise to a subset if requested
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
    
    return new_dataset, labeled_indices

def build_constraints_from_partial_data(dataset, must_link_mode="same_label", 
                                      cannot_link_mode="diff_label", 
                                      max_pairs=1000, 
                                      cannot_link_fraction=0.1,  # NEW: only use 10% of cannot-link
                                      seed=42):
    """
    Create pairwise constraints from a partially labeled dataset.
    Args:
        dataset: list of (image, label), where label == -1 means unlabeled
        must_link_mode (str): "same_label" => any pair with same label becomes must-link
        cannot_link_mode (str): "diff_label" => any pair with different label => cannot-link
        max_pairs (int): how many pairs to sample to keep constraints manageable
        cannot_link_fraction (float): fraction of possible cannot-link constraints to use
        seed (int): random seed
    Returns:
        constraints: dict with 'must_link', 'cannot_link' lists of (i, j) pairs
    """
    import random
    rng = random.Random(seed)
    labeled_indices = [i for i, (_, lbl) in enumerate(dataset) if lbl != -1]
    
    must_link_pairs = []
    cannot_link_pairs = []

    # Group labeled samples by label
    from collections import defaultdict
    label_to_indices = defaultdict(list)
    for i in labeled_indices:
        label = dataset[i][1]
        label_to_indices[label].append(i)

    # Build must_link for samples with same label (keep all of these)
    for lbl, indices in label_to_indices.items():
        if len(indices) > 1:
            # sample pairs among these indices
            all_pairs = []
            for i1 in range(len(indices)):
                for i2 in range(i1+1, len(indices)):
                    all_pairs.append((indices[i1], indices[i2]))
            # shuffle & possibly limit number
            rng.shuffle(all_pairs)
            must_link_pairs.extend(all_pairs[:max_pairs])

    # Build cannot_link for samples with different labels
    # But only use a fraction of possible cannot-link constraints
    all_labels = list(label_to_indices.keys())
    if len(all_labels) > 1:
        all_cannot_links = []
        for i in range(len(all_labels)):
            for j in range(i+1, len(all_labels)):
                lbl_i = all_labels[i]
                lbl_j = all_labels[j]
                # all cross pairs
                for id_i in label_to_indices[lbl_i]:
                    for id_j in label_to_indices[lbl_j]:
                        all_cannot_links.append((id_i, id_j))
        
        # Randomly sample a fraction of cannot-link constraints
        rng.shuffle(all_cannot_links)
        n_cannot_links = min(
            int(len(all_cannot_links) * cannot_link_fraction),
            max_pairs
        )
        cannot_link_pairs = all_cannot_links[:n_cannot_links]

    if len(must_link_pairs) > 0 or len(cannot_link_pairs) > 0:
        print(f"Generated {len(must_link_pairs)} must-link and {len(cannot_link_pairs)} cannot-link constraints")
        print(f"Ratio cannot-link/must-link: {len(cannot_link_pairs)/max(1, len(must_link_pairs)):.2f}")

    constraints = {
        'must_link': must_link_pairs,
        'cannot_link': cannot_link_pairs
    }
    return constraints


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
