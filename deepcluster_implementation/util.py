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



def create_sparse_labels(dataset, fraction=0.1, pattern="random", noise=0.0, seed=42):
    """
    Create a partially labeled dataset from a fully labeled dataset.
    
    Args:
        dataset: a PyTorch Dataset where dataset[i] = (image, label)
        fraction (float): fraction of labels we want to *keep* (e.g., 0.1 means 10%).
        pattern (str): "random", "missing_classes", or custom logic
        noise (float): fraction of kept labels that we corrupt (0.0 means no noise)
        seed (int): random seed for reproducibility

    Returns:
        new_dataset: a list or custom dataset of (image, label) where label==-1 means unlabeled
    """
    rng = random.Random(seed)
    total_size = len(dataset)

    # Convert dataset to list for easy manipulation
    data_list = [(dataset[i][0], dataset[i][1]) for i in range(total_size)]
    
    if pattern == "random":
        # random subset of labeled indices
        n_labeled = int(fraction * total_size)
        labeled_indices = rng.sample(range(total_size), n_labeled)
        labeled_indices = set(labeled_indices)
    elif pattern == "missing_classes":
        # e.g. remove all labels for half the classes
        # This is just an example; adapt as needed.
        # Let's find unique classes:
        all_labels = [label for (_, label) in data_list]
        classes = list(set(all_labels))
        rng.shuffle(classes)
        half = len(classes) // 2
        unlabeled_classes = set(classes[:half])
        
        labeled_indices = []
        for i, (img, lbl) in enumerate(data_list):
            if lbl not in unlabeled_classes:
                labeled_indices.append(i)
        labeled_indices = set(labeled_indices)
        
        # You could also keep fraction within those classes, etc.
    else:
        # fallback or custom pattern
        labeled_indices = set(range(total_size))  # by default label all
    
    # Build new dataset with partial labels
    new_dataset = []
    for i, (img, lbl) in enumerate(data_list):
        if i not in labeled_indices:
            # remove label
            new_dataset.append((img, -1))
        else:
            # keep label
            new_dataset.append((img, lbl))
    
    # Introduce noise in the kept labels (optional)
    if noise > 0.0:
        # the portion of labeled samples we will corrupt
        # e.g., noise=0.1 means 10% of the labeled samples get assigned random label
        labeled_inds_list = list(labeled_indices)
        rng.shuffle(labeled_inds_list)
        n_noisy = int(noise * len(labeled_inds_list))
        noisy_part = labeled_inds_list[:n_noisy]
        all_possible_labels = list(set([l for (x, l) in data_list]))
        for idx in noisy_part:
            # pick a random label from all_possible_labels (could exclude the true one if desired)
            corrupted_lbl = rng.choice(all_possible_labels)
            new_dataset[idx] = (new_dataset[idx][0], corrupted_lbl)

    return new_dataset

def build_constraints_from_partial_data(dataset, must_link_mode="same_label", cannot_link_mode="diff_label", max_pairs=1000, seed=42):
    """
    Create pairwise constraints from a partially labeled dataset.
    Args:
        dataset: list of (image, label), where label == -1 means unlabeled
        must_link_mode (str): "same_label" => any pair with same label becomes must-link
        cannot_link_mode (str): "diff_label" => any pair with different label => cannot-link
        max_pairs (int): how many pairs to sample to keep constraints manageable
        seed (int): random seed
    Returns:
        constraints: dict with 'must_link', 'cannot_link' lists of (i, j) pairs
    """
    import random
    rng = random.Random(seed)
    labeled_indices = [i for i, (_, lbl) in enumerate(dataset) if lbl != -1]
    
    must_link_pairs = []
    cannot_link_pairs = []

    # Simple approach: group labeled samples by label
    from collections import defaultdict
    label_to_indices = defaultdict(list)
    for i in labeled_indices:
        label = dataset[i][1]
        label_to_indices[label].append(i)

    # Build must_link for samples with same label
    for lbl, indices in label_to_indices.items():
        if len(indices) > 1:
            # sample pairs among these indices
            all_pairs = []
            for i1 in range(len(indices)):
                for i2 in range(i1+1, len(indices)):
                    all_pairs.append((indices[i1], indices[i2]))
            # shuffle & possibly limit number
            rng.shuffle(all_pairs)
            must_link_pairs.extend(all_pairs[:max_pairs])  # or you can do some fraction

    # Build cannot_link for samples with different labels
    # This can grow huge. We'll randomly sample across labels
    all_labels = list(label_to_indices.keys())
    if len(all_labels) > 1:
        for i in range(len(all_labels)):
            for j in range(i+1, len(all_labels)):
                lbl_i = all_labels[i]
                lbl_j = all_labels[j]
                # all cross pairs
                cross_pairs = []
                for id_i in label_to_indices[lbl_i]:
                    for id_j in label_to_indices[lbl_j]:
                        cross_pairs.append((id_i, id_j))
                rng.shuffle(cross_pairs)
                cannot_link_pairs.extend(cross_pairs[:max_pairs])

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
