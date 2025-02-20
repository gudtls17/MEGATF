from source.utils import accuracy, TotalMeter, count_params, isfloat, generate_within_brain_negatives
import torch
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support, classification_report
from source.utils import continus_mixup_data
import wandb
from omegaconf import DictConfig
from typing import List
import torch.utils.data as utils
from source.components import LRScheduler
import logging
import pickle
import time


class Train:

    def __init__(self, cfg: DictConfig,
                 model: torch.nn.Module,
                 optimizers: List[torch.optim.Optimizer],
                 lr_schedulers: List[LRScheduler],
                 dataloaders: List[utils.DataLoader],
                 logger: logging.Logger,
                 foldername) -> None:

        self.config = cfg
        self.logger = logger
        self.model = model
        self.logger.info(f'#model params: {count_params(self.model)}')
        self.train_dataloader, self.test_dataloader = dataloaders
        self.epochs = cfg.training.epochs
        self.total_steps = cfg.total_steps
        self.optimizers = optimizers
        self.lr_schedulers = lr_schedulers
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
        self.save_path = Path(cfg.log_path) / cfg.unique_id / foldername
        self.save_learnable_graph = cfg.save_learnable_graph
        self.save_attn_weights = cfg.save_attn_weights
        self.save_test_attn_weights = cfg.save_test_attn_weights
        

        self.init_meters()
        
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        non_trainable_params = total_params - trainable_params
        print("Total params : ", total_params)
        print("Trainable params : ", trainable_params)
        print("Non-trainable params : ", non_trainable_params)
        
        print("Train :", len(self.train_dataloader))
        print("Test :", len(self.test_dataloader))

    def init_meters(self):
        self.train_loss, self.val_loss,\
            self.test_loss, self.train_accuracy,\
            self.val_accuracy, self.test_accuracy = [
                TotalMeter() for _ in range(6)]

    def reset_meters(self):
        for meter in [self.train_accuracy, self.val_accuracy,
                      self.test_accuracy, self.train_loss,
                      self.val_loss, self.test_loss]:
            meter.reset()

    def train_per_epoch(self, optimizer, lr_scheduler):
        self.model.train()
        for batched_graph, label in self.train_dataloader:
            label = label.float()
            self.current_step += 1

            lr_scheduler.update(optimizer=optimizer, step=self.current_step)
            
            """Node feature"""
            batch_size = batched_graph.batch_size
            node_feature = batched_graph.ndata['x'] 
            edge_feature1 = batched_graph.edata['e1']
            edge_feature2 = batched_graph.edata['e2']
            
            node_feature = node_feature.reshape(batch_size, -1, self.config.dataset.node_feature_sz)
            edge_feature1 = edge_feature1.reshape(batch_size, -1, 1)
            edge_feature2 = edge_feature2.reshape(batch_size, -1, 1)
            
            if self.config.preprocess.continus:
                node_feature, edge_feature1, edge_feature2, label = continus_mixup_data(
                    node_feature.cuda(), edge_feature1.cuda(), edge_feature2.cuda(), y=label.cuda())
            
            node_feature = node_feature.reshape(-1, self.config.dataset.node_feature_sz)
            edge_feature1 = edge_feature1.reshape(-1, 1)
            edge_feature2 = edge_feature2.reshape(-1, 1)
            
            edge_feature = torch.concat((edge_feature1, edge_feature2), 1)  # main
            
            
            """Training"""
            batched_graph = batched_graph.to(torch.device("cuda:0"))
            node_feature, edge_feature, label = node_feature.cuda(), edge_feature.cuda(), label.cuda()
            predict, node_feature_last_layer  = self.model(batched_graph, node_feature, edge_feature, batch_size) 
            
            classifier_loss = self.loss_fn(predict, label)
            
            loss = classifier_loss

            self.train_loss.update_with_weight(loss.item(), label.shape[0])
            
            optimizer.zero_grad()
            loss.backward()
            
            optimizer.step()
            top1 = accuracy(predict, label[:, 1])[0]
            self.train_accuracy.update_with_weight(top1, label.shape[0])

    def test_per_epoch(self, dataloader, loss_meter, acc_meter):
        labels = []
        result = []

        self.model.eval()

        for batched_graph, label in dataloader:
            
            """Node feature"""
            batch_size = batched_graph.batch_size
            node_feature = batched_graph.ndata['x']
            edge_feature1 = batched_graph.edata['e1']
            edge_feature2 = batched_graph.edata['e2']
            
            edge_feature = torch.concat((edge_feature1, edge_feature2), 1)  # main
            
            """Test"""
            batched_graph = batched_graph.to(torch.device("cuda:0"))
            node_feature, edge_feature, label = node_feature.cuda(), edge_feature.cuda(), label.cuda()
            output, node_feature_last_layer  = self.model(batched_graph, node_feature, edge_feature, batch_size)  

            label = label.float()

            classifier_loss = self.loss_fn(output, label)
            
            loss = classifier_loss
            
            loss_meter.update_with_weight(
                loss.item(), label.shape[0])
            top1 = accuracy(output, label[:, 1])[0]
            acc_meter.update_with_weight(top1, label.shape[0])
            result += F.softmax(output, dim=1)[:, 1].tolist()
            labels += label[:, 1].tolist()

        auc = roc_auc_score(labels, result)
        result, labels = np.array(result), np.array(labels)
        result[result > 0.5] = 1
        result[result <= 0.5] = 0
        metric = precision_recall_fscore_support(
            labels, result, average='micro')

        report = classification_report(
            labels, result, output_dict=True, zero_division=0)

        recall = [0, 0]
        for k in report:
            if isfloat(k):
                recall[int(float(k))] = report[k]['recall']
        return [auc] + list(metric) + recall

    def save_attention_weights(self):
        attn_weights = []
        labels = []
        assign_matrices = []
        self.model.eval()
        for batched_graph, label in self.train_dataloader:
            
            """Node feature"""
            batch_size = batched_graph.batch_size
            node_feature = batched_graph.ndata['x']
            edge_feature1 = batched_graph.edata['e1']
            edge_feature2 = batched_graph.edata['e2']
            
            edge_feature = torch.concat((edge_feature1, edge_feature2), 1)  # main
            
            """Model"""
            batched_graph = batched_graph.to(torch.device("cuda:0"))
            node_feature, edge_feature, label = node_feature.cuda(), edge_feature.cuda(), label.cuda()
            prediction, _  = self.model(batched_graph, node_feature, edge_feature, batch_size)  

            assignMat = self.model.get_assign_mat()
            assign_np = assignMat.detach().cpu().numpy()
            assign_matrices.append(assign_np)

            # attn = self.model.get_attention_weights()
            # attn_np = attn[0].detach().cpu().numpy()
            label_np = label.detach().cpu().numpy()
            labels.append(label_np)
            # attn_weights.append(attn_np)


        if self.save_test_attn_weights:
            attn_weights_test = []
            labels_test = []
            assign_matrices_test = []     
            
            node_feature_last_layer_test = []
            

        for batched_graph, label in self.test_dataloader:
            
            """Node feature"""
            batch_size = batched_graph.batch_size
            node_feature = batched_graph.ndata['x']
            edge_feature1 = batched_graph.edata['e1']
            edge_feature2 = batched_graph.edata['e2']
            
            edge_feature = torch.concat((edge_feature1, edge_feature2), 1)  # main
            
            """Model"""
            batched_graph = batched_graph.to(torch.device("cuda:0"))
            node_feature, edge_feature, label = node_feature.cuda(), edge_feature.cuda(), label.cuda()
            prediction, node_feature_last_layer = self.model(batched_graph, node_feature, edge_feature, batch_size) 

            assignMat = self.model.get_assign_mat()
            assign_np = assignMat.detach().cpu().numpy()
            assign_matrices.append(assign_np)


            # attn = self.model.get_attention_weights()
            # attn_np = attn[0].detach().cpu().numpy()
            label_np = label.detach().cpu().numpy()
            labels.append(label_np)
            # attn_weights.append(attn_np)

            if self.save_test_attn_weights:
                assign_matrices_test.append(assign_np)
                labels_test.append(label_np)
                # attn_weights_test.append(attn_np)
                
                node_feature_last_layer_test.append(node_feature_last_layer.detach().cpu().numpy())
                

        # np.save(self.save_path/f"attnWeights.npy", attn_weights, allow_pickle=True)
        np.save(self.save_path/f"labels.npy", np.concatenate(labels), allow_pickle=True)
        np.save(self.save_path/f"assign_matrices.npy", np.concatenate(assign_matrices), allow_pickle=True)

        if self.save_test_attn_weights:
            # np.save(self.save_path/f"attnWeights_test.npy", attn_weights_test, allow_pickle=True)
            np.save(self.save_path/f"labels_test.npy", np.concatenate(labels_test), allow_pickle=True)
            np.save(self.save_path/f"assign_matrices_test.npy", np.concatenate(assign_matrices_test), allow_pickle=True)
            
            np.save(self.save_path/f"node_feature_last_layer_test.npy", np.concatenate(node_feature_last_layer_test), allow_pickle=True)

    def generate_save_learnable_matrix(self):

        learable_matrixs = []

        labels = []

        for batched_graph, label in self.test_dataloader:
            label = label.long()
            
            """Node feature"""
            batch_size = batched_graph.batch_size
            node_feature = batched_graph.ndata['x']
            edge_feature1 = batched_graph.edata['e1']
            edge_feature2 = batched_graph.edata['e2']
            
            edge_feature = torch.concat((edge_feature1, edge_feature2), 1)  # main
            
            """Model"""
            batched_graph = batched_graph.to(torch.device("cuda:0"))
            node_feature, edge_feature, label = node_feature.cuda(), edge_feature.cuda(), label.cuda()
            learable_matrix, _ = self.model(batched_graph, node_feature, edge_feature, batch_size) 

            learable_matrixs.append(learable_matrix.cpu().detach().numpy())
            labels += label.tolist()

        self.save_path.mkdir(exist_ok=True, parents=True)
        np.save(self.save_path/"learnable_matrix.npy", {'matrix': np.vstack(
            learable_matrixs), "label": np.array(labels)}, allow_pickle=True)

    def save_result(self, results: torch.Tensor):
        self.save_path.mkdir(exist_ok=True, parents=True)
        np.save(self.save_path/"training_process.npy",
                results, allow_pickle=True)

        torch.save(self.model.state_dict(), self.save_path/"model.pt")

    def train(self):
        training_process = []
        self.current_step = 0
        best_val_AUC = 0
        best_test_acc = 0
        best_test_AUC = 0
        best_test_sen = 0
        best_test_spec = 0
        best_model = None
        for epoch in range(self.epochs):
            self.reset_meters()
            self.train_per_epoch(self.optimizers[0], self.lr_schedulers[0])
            test_result = self.test_per_epoch(self.test_dataloader,
                                              self.test_loss, self.test_accuracy)
            # raise

            self.logger.info(" | ".join([
                f'Epoch[{epoch}/{self.epochs}]',
                f'Train Loss:{self.train_loss.avg: .3f}',
                f'Train Accuracy:{self.train_accuracy.avg: .3f}%',
                f'Test Loss:{self.test_loss.avg: .3f}',
                f'Test Accuracy:{self.test_accuracy.avg: .3f}%',
                f'Test AUC:{test_result[0]:.4f}',
                f'Test Sen:{test_result[-1]:.4f}',
                f'Test Spec:{test_result[-2]:.4f}',
                f'LR:{self.lr_schedulers[0].lr:.5f}'
            ]))

            wandb.log({
                "Train Loss": self.train_loss.avg,
                "Train Accuracy": self.train_accuracy.avg,
                "Test Loss": self.test_loss.avg,
                "Test Accuracy": self.test_accuracy.avg,
                "Test AUC": test_result[0],
                'Test Sensitivity': test_result[-1],
                'Test Specificity': test_result[-2],
                'micro F1': test_result[-4],
                'micro recall': test_result[-5],
                'micro precision': test_result[-6],
            })

            if(test_result[0] > best_test_AUC):
                best_test_acc =  self.test_accuracy.avg
                best_test_AUC =  test_result[0]
                best_test_sen = test_result[-1]
                best_test_spec = test_result[-2]
                wandb.run.summary["Best Test Accuracy"] = self.test_accuracy.avg
                wandb.run.summary["Best Test AUC"] = test_result[0]
                wandb.run.summary["Best Test Sensitivity"] = test_result[-1]
                wandb.run.summary["Best Test Specificity"] = test_result[-2]
                
                best_model = self.model.state_dict()
                if self.save_attn_weights:
                    self.save_attention_weights()

                if self.save_learnable_graph:
                    self.generate_save_learnable_matrix()
                torch.save(best_model, self.save_path/"best_model.pt")
                print("Best Model Updated")
                
            training_process.append({
                "Epoch": epoch,
                "Train Loss": self.train_loss.avg,
                "Train Accuracy": self.train_accuracy.avg,
                "Test Loss": self.test_loss.avg,
                "Test Accuracy": self.test_accuracy.avg,
                "Test AUC": test_result[0],
                'Test Sensitivity': test_result[-1],
                'Test Specificity': test_result[-2],
                'micro F1': test_result[-4],
                'micro recall': test_result[-5],
                'micro precision': test_result[-6],
            })
            

        self.save_result(training_process)
        
        return [best_test_acc,best_test_AUC,best_test_sen,best_test_spec]
