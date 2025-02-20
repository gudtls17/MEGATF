import torch
import torch.utils.data as utils
from omegaconf import DictConfig, open_dict
from typing import List
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import torch.nn.functional as F

from dgl.dataloading import GraphDataLoader
from .adni import ADNI_Dataset



def init_dataloader(cfg: DictConfig,
                    final_pearson: torch.tensor,
                    final_timedelay: torch.tensor,
                    final_its: torch.tensor,
                    labels: torch.tensor) -> List[utils.DataLoader]:
    labels = F.one_hot(labels.to(torch.int64))
    length = final_pearson.shape[0]
    train_length = int(length*cfg.dataset.train_set*cfg.datasz.percentage)
    val_length = int(length*cfg.dataset.val_set)
    if cfg.datasz.percentage == 1.0:
        test_length = length-train_length-val_length
    else:
        test_length = int(length*(1-cfg.dataset.val_set-cfg.dataset.train_set))

    with open_dict(cfg):
        # total_steps, steps_per_epoch for lr schedular
        cfg.steps_per_epoch = (
            train_length - 1) // cfg.dataset.batch_size + 1
        cfg.total_steps = cfg.steps_per_epoch * cfg.training.epochs

    dataset = utils.TensorDataset(
        final_pearson[:train_length+val_length+test_length],
        final_timedelay[:train_length+val_length+test_length],
        final_its[:train_length+val_length+test_length],
        labels[:train_length+val_length+test_length]
    )

    train_dataset, val_dataset, test_dataset = utils.random_split(
        dataset, [train_length, val_length, test_length])
    train_dataloader = utils.DataLoader(
        train_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=cfg.dataset.drop_last)

    val_dataloader = utils.DataLoader(
        val_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=False)

    test_dataloader = utils.DataLoader(
        test_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=False)

    return [train_dataloader, val_dataloader, test_dataloader]


def init_stratified_dataloader(cfg: DictConfig,
                               split_index,
                               graphs,
                               labels):
    train_index, test_index = split_index
    labels = F.one_hot(labels.to(torch.int64))

    graphs_train, labels_train = graphs[train_index], labels[train_index]
    graphs_test, labels_test = graphs[test_index], labels[test_index]
    
    train_dataset = ADNI_Dataset(graphs_train, labels_train)
    test_dataset = ADNI_Dataset(graphs_test, labels_test)
    
    train_dataloader = GraphDataLoader(train_dataset, batch_size=cfg.dataset.batch_size, drop_last=False, shuffle=True)
    test_dataloader = GraphDataLoader(test_dataset, batch_size=cfg.dataset.batch_size, drop_last=False, shuffle=True)

    return [train_dataloader, test_dataloader]

def init_stratified_crossvalidation(cfg: DictConfig,
                               graphs,
                               labels: torch.tensor,):
    
    length = len(graphs)
    train_length = int(length*0.8)
    test_length = length - train_length

    with open_dict(cfg):
        # total_steps, steps_per_epoch for lr schedular
        cfg.steps_per_epoch = (
            train_length - 1) // cfg.dataset.batch_size + 1
        cfg.total_steps = cfg.steps_per_epoch * cfg.training.epochs

    train_test_index_list = []
    dataloader_list = []
    split = StratifiedShuffleSplit(n_splits=5, train_size=train_length, test_size=test_length, random_state=42)
    for train_index, test_index in split.split(graphs, labels):
        train_test_index_list.append([train_index, test_index])

    return train_test_index_list