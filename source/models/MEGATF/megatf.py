from os.path import join
import json
import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer
from .ptdec import DEC
from typing import List
from .components import GraphTransformerEncoder
from omegaconf import DictConfig
from ..base import BaseModel
import pickle

class GraphTransforemrPoolingEncoder(nn.Module):
    """
    GraphTransformer encoder with Pooling mechanism.
    Input size: (batch_size, input_node_num, input_feature_size)
    Output size: (batch_size, output_node_num, input_feature_size)
    """

    def __init__(self, input_feature_size, input_node_num, hidden_size, output_node_num, pooling=True, orthogonal=True, freeze_center=False, project_assignment=True, nHead=2):
        super().__init__()
        self.transformer = GraphTransformerEncoder(hidden_dim=input_feature_size, n_classes=2, nheads=nHead, dim_feedforward=hidden_size, mha_conv="EdgeGAT")  # EdgeGAT  GAT
        
        self.pooling = pooling
        if self.pooling:
            encoder_hidden_size = 32
            self.encoder = nn.Sequential(
                nn.Linear(input_feature_size *
                          input_node_num, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size,
                          input_feature_size * input_node_num),
            )
            self.dec = DEC(cluster_number=output_node_num, hidden_dimension=input_feature_size, encoder=self.encoder,
                           orthogonal=orthogonal, freeze_center=freeze_center, project_assignment=project_assignment)
        
        
    def is_pooling_enabled(self):
        return self.pooling

    def forward(self, batched_graph, feats, efeats, batch_size):
        
        x_last_layer = self.transformer(batched_graph, feats, efeats, batch_size)  # batch, input_node_num, input_feature_size 
        
        if self.pooling:
            x, assignment = self.dec(x_last_layer)
            return x, assignment, x_last_layer
        return x_last_layer, None, x_last_layer

    def get_attention_weights(self):
        return self.transformer.get_attention_weights()

    def loss(self, assignment):
        return self.dec.loss(assignment)


class Megatf(BaseModel):
    def __init__(self, config: DictConfig):

        super().__init__()
        
        self.attention_list = nn.ModuleList()
        # forward_dim = config.dataset.node_sz
        forward_dim = config.dataset.node_feature_sz

        self.pos_encoding = config.model.pos_encoding
        if self.pos_encoding == 'identity':
            self.node_identity = nn.Parameter(torch.zeros(
                config.dataset.node_sz, config.model.pos_embed_dim), requires_grad=True)
            forward_dim = config.dataset.node_sz + config.model.pos_embed_dim
            nn.init.kaiming_normal_(self.node_identity)

        self.num_MHSA = config.model.num_MHSA
        sizes = config.model.sizes
        sizes[0] = config.dataset.node_sz
        in_sizes = [config.dataset.node_sz] + sizes[:-1]
        do_pooling = config.model.pooling
        self.do_pooling = do_pooling 

        self.attention_list.append(GraphTransforemrPoolingEncoder(input_feature_size=forward_dim, input_node_num=in_sizes[1], hidden_size=1024, output_node_num=sizes[1],
                                                               pooling=True, orthogonal=config.model.orthogonal, freeze_center=config.model.freeze_center,
                                                               project_assignment=config.model.project_assignment, nHead=config.model.nhead))
        self.dim_reduction = nn.Sequential(nn.Linear(forward_dim, 8), nn.LeakyReLU())
        self.fc = nn.Sequential(
            nn.Linear(8 * sizes[-1], 256),
            nn.LeakyReLU(),
            nn.Linear(256, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 2)
        )
        
        self.assignMat = None

        
    def forward(self, batched_graph, feats, efeats, batch_size):
        """
        input: batched graph
               
        node feature: input node feature (batch*input_node_num, input_feature_size)
        edge feature: input edge feature (batch*input_node_num, input_feature_size, 2)
        
        Batched_graph -> 

        """
        
        if self.pos_encoding == 'identity':
            pos_emb = self.node_identity.expand(batch_size, *self.node_identity.shape)
            connectivity = torch.cat([connectivity, pos_emb], dim=-1)

        assignments = []
        attn_weights = []
        
        node_feature, assign, node_feature_last_layer = self.attention_list[0](batched_graph, feats, efeats, batch_size) # graph transforemr + OCReadout
        assignments.append(assign)
        attn_weights.append(self.attention_list[0].get_attention_weights())
        
        self.assignMat = assignments[0]
        
        node_feature = self.dim_reduction(node_feature)  # (batch, cluster_num, input_feature_size) -> (batch, cluster_num, 8)
        node_feature = node_feature.reshape((batch_size, -1))  # (batch, cluster_num, 8) -> (batch, cluster_num*8)
        
        return self.fc(node_feature), node_feature_last_layer

    def get_assign_mat(self):
        return self.assignMat

    def get_attention_weights(self):
        return [atten.get_attention_weights() for atten in self.attention_list]

    def get_local_attention_weights(self):
        return self.local_transformer.get_attention_weights()

    def get_cluster_centers(self) -> torch.Tensor:
        """
        Get the cluster centers, as computed by the encoder.

        :return: [number of clusters, hidden dimension] Tensor of dtype float
        """
        return self.dec.get_cluster_centers()

    def loss(self, assignments):
        """
        Compute KL loss for the given assignments. Note that not all encoders contain a pooling layer.
        Inputs: assignments: [batch, number of clusters]
        Output: KL loss
        """
        decs = list(
            filter(lambda x: x.is_pooling_enabled(), self.attention_list))
        assignments = list(filter(lambda x: x is not None, assignments))
        loss_all = None

        for index, assignment in enumerate(assignments):
            if loss_all is None:
                loss_all = decs[index].loss(assignment)
            else:
                loss_all += decs[index].loss(assignment)
        return loss_all


