from omegaconf import DictConfig, open_dict
from .adni import load_adni_data, load_adni_MCI_data, load_hbn_data
from .dataloader import init_dataloader, init_stratified_dataloader, init_stratified_crossvalidation
from typing import List
import torch.utils as utils


def dataset_factory(cfg: DictConfig, split_index) -> List[utils.data.DataLoader]:

    assert cfg.dataset.name in ['adni', 'adni_MCI', 'hbn']

    datasets = eval(
        f"load_{cfg.dataset.name}_data")(cfg)

    dataloaders = init_stratified_dataloader(cfg, split_index, *datasets) \
        if cfg.dataset.stratified \
        else init_dataloader(cfg, *datasets)

    return dataloaders



def split_factory(cfg: DictConfig) -> List[utils.data.DataLoader]:

    assert cfg.dataset.name in ['adni', 'adni_MCI', 'hbn']

    datasets = eval(
        f"load_{cfg.dataset.name}_data")(cfg)

    split_index = init_stratified_crossvalidation(cfg, *datasets) \

    return split_index