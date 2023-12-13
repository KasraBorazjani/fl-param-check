import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset

from typing import Sequence
import numpy as np

def create_ds_from_labels(primary_label:int,
                          secondary_labels:Sequence[int],
                          label_indexes_list,
                          total_ds_len:int,
                          primary_label_fraction:float,
                          original_ds:Dataset
                          ):
    
    
    # Find the primary label elements for dataset
    primary_label_elements_num = int(primary_label_fraction * total_ds_len)
    # print("num prim elems: ", primary_label_elements_num)
    selected_primary_indexes = np.random.randint(low=0, high=len(label_indexes_list[primary_label]), size=primary_label_elements_num)
    # print(selected_primary_indexes)
    primary_label_idxs = label_indexes_list[primary_label][selected_primary_indexes]
    label_indexes_list[primary_label] = label_indexes_list[primary_label][~np.isin(label_indexes_list[primary_label], primary_label_idxs)]
    primary_label_subset = Subset(original_ds, primary_label_idxs)
    # print("len subset prim: ", len(primary_label_subset))
    
    # Find the secondary label(s) elements for the dataset
    secondary_label_elements_frac = ((1-primary_label_fraction)/len(secondary_labels))
    secondary_label_elements_num = int(secondary_label_elements_frac*total_ds_len)
    secondary_labels_subsets = []
    for label in secondary_labels:
        selected_label_indexes = np.random.randint(low=0, high=len(label_indexes_list[label]), size=secondary_label_elements_num)
        selected_indexes = label_indexes_list[label][selected_label_indexes]
        secondary_labels_subsets.append(Subset(original_ds, selected_indexes))
        label_indexes_list[label] = label_indexes_list[label][~np.isin(label_indexes_list[label], selected_indexes)]
    
    # for item in secondary_labels_subsets:
    #     print(len(item))

    secondary_labels_subsets += [primary_label_subset]

    return ConcatDataset(secondary_labels_subsets), label_indexes_list  

def load_original_dataset(data_path):

    mnist_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
    
    train_set = torchvision.datasets.MNIST(data_path, download=True, train=True, transform=mnist_transforms)
    test_set = torchvision.datasets.MNIST(data_path, download=True, train=False, transform=mnist_transforms)

    # "10" is the number of labels we have in MNIST dataset (handwritten digits)
    label_idxs = [np.array([]) for i in range(10)]

    for i, datapoint in enumerate(train_set):
        label_idxs[datapoint[1]] = np.append(label_idxs[datapoint[1]], i)
    
    return train_set, test_set, label_idxs

def has_enough_samples(label_idx_list, n_labels_needed, label_chosen):
    if len(label_idx_list[label_chosen]) >= n_labels_needed:
        return True
    else:
        return False

def generate_random_label_set(label_idx_list, primary_dataset_len, secondary_dataset_len, num_secondaries):
    selected_labels = []
    while True:
        sample_label = np.random.randint(low=0, high=10, dtype=int)
        if has_enough_samples(label_idx_list, primary_dataset_len, sample_label):
            selected_labels.append(sample_label)
            break
    
    for i in range(num_secondaries):
        while True:
            sample_label = np.random.randint(low=0, high=10, dtype=int)
            if (has_enough_samples(label_idx_list, primary_dataset_len, sample_label)) and (sample_label not in selected_labels):
                selected_labels.append(sample_label)
                break
        
    return selected_labels

def create_client_ds(original_ds, label_idx_list, total_ds_len, primary_label_fraction, num_secondaries):
    primary_dataset_len = int(primary_label_fraction * total_ds_len)
    secondary_dataset_len = int(((1-primary_label_fraction)/num_secondaries) * total_ds_len)
    label_set = generate_random_label_set(label_idx_list, primary_dataset_len, secondary_dataset_len, num_secondaries)
    client_ds, label_idx_list = create_ds_from_labels(label_set[0], label_set[1:], label_idx_list,
                                total_ds_len, primary_label_fraction, original_ds)
    
    return client_ds, label_idx_list, label_set
