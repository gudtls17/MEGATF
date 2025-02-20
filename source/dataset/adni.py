import numpy as np
import torch
from .preprocess import StandardScaler
from omegaconf import DictConfig, open_dict
from dgl.data import DGLDataset

def load_adni_data(cfg: DictConfig):
    
    data = np.load(cfg.dataset.path, allow_pickle=True).item()
    graphs = data["Graph"]
    labels = data["label"]   # 0: CN, 1: MCI
    
    labels = torch.from_numpy(labels).float()

    with open_dict(cfg):

        cfg.dataset.node_sz, cfg.dataset.node_feature_sz = graphs[0].ndata['x'].shape

    return graphs, labels

def load_adni_MCI_data(cfg: DictConfig):
    
    data = np.load(cfg.dataset.path, allow_pickle=True).item()
    graphs = data["Graph"]
    labels = data["label"]   # 0: eMCI, 1: lMCI
    
    labels = torch.from_numpy(labels).float()

    with open_dict(cfg):

        cfg.dataset.node_sz, cfg.dataset.node_feature_sz = graphs[0].ndata['x'].shape

    return graphs, labels

def load_hbn_data(cfg: DictConfig):
    
    data = np.load(cfg.dataset.path, allow_pickle=True).item()
    graphs = data["Graph"]
    labels = data["label"]   # 0: CN, 1: ADHD
    
    labels = torch.from_numpy(labels).float()

    with open_dict(cfg):

        cfg.dataset.node_sz, cfg.dataset.node_feature_sz = graphs[0].ndata['x'].shape

    return graphs, labels

class ADNI_Dataset(DGLDataset):
    def __init__(self, graphs, label_list,):
        super().__init__(name="ADNI_CN_MCI")
        
        # get data variable
        self.label_list = label_list
        
        # make list
        self.graphs = []
        self.labels = []
        
        # define graph structure of each data
        for i in range(len(graphs)):
            label = self.label_list[i]
            g = graphs[i]
            self.graphs.append(g)
            self.labels.append(label)
        
        # Convert the label list to tensor for saving.
        # self.labels = torch.LongTensor(self.labels)

    def process(self):
        return None

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)