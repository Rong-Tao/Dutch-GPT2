import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
# dataset.py
from datasets import load_from_disk

def get_dataset():
    dataset = load_from_disk("../../dataset/Tokenized_dataset.hf")
    return dataset


def get_loaders(world_size, rank, batch_size, split_ratio):
    full_dataset = get_dataset()
    full_dataset.set_format('torch')
    train_size = int(split_ratio * len(full_dataset))
    validation_size = len(full_dataset) - train_size
    train_dataset, validation_dataset = random_split(full_dataset, [train_size, validation_size])

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)

    validation_sampler = DistributedSampler(validation_dataset, num_replicas=world_size, rank=rank)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, sampler=validation_sampler)

    return train_loader, validation_loader