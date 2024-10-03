import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import TanhTransform

from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
import torch_scatter

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import torch.optim as optim

from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier

class BaselineLinear(pl.LightningModule):
    def __init__(self, architecture, n_features, n_output_dim_node, n_output_dim_action, 
                    loss_module=nn.MSELoss(), split_mode="", lr=0.1, limit_train=False, dropout_p=0.5):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = lr
        
        self.model = torch.nn.Sequential()
        layers = architecture.split('-')
        #layers.append(n_output_dim_node)
        input_dim = n_features
        for i in range(len(layers)):
            self.model.add_module("Lin"+str(i), nn.Linear(input_dim, int(layers[i])))
            self.model.add_module("Relu"+str(i), nn.ReLU())
            self.model.add_module("Dropout"+str(i), nn.Dropout(dropout_p))
            input_dim = int(layers[i])
        self.model.add_module("Lin"+str(len(layers)), nn.Linear(input_dim, int(n_output_dim_node)))
        self.softmax = nn.Softmax(dim=1)
        self.loss_module = loss_module
        self.num_parameters = self.count_parameters(self )
        self.print_parameters()

    def print_parameters(self):
        print('Model:', self.count_parameters(self.model ), self.model)
        print('Overall:', self.count_parameters(self )) 

    def count_parameters(self, model): 
        return sum(p.numel() for p in model.parameters() if p.requires_grad) 

    def forward(self, data, mode="train"):
        x, batch_idx = data.x, data.batch
        
        num_nodes = torch.Tensor(np.array(data.num_nodes_loc)).to(torch.int64).to(device=x.device).squeeze(1)
        
        y_out = self.softmax(self.model(x))

        loss = self.loss_module(y_out,data.y_who_one.clone().detach().float()) 
        
        preds_val, preds_batch_id = torch_scatter.scatter_max(y_out[:,0], batch_idx, dim=0)

        target_ids_batches = []
        for n in num_nodes:
            target_ids_batches.extend(np.arange(n))
        
        index_preds_batch = np.array(preds_batch_id.squeeze()).astype(int)
        preds = torch.tensor(np.array(target_ids_batches)[index_preds_batch])
   
        try:
            acc = (preds == data.y_who).sum().float() / preds.shape[0]
        except Exception as  e:
            try:
                #print(e)
                # this case is a fall back in case there is only one group in the batch
                # in this case preds is a single number and not an array
                preds = torch.Tensor([preds])

                acc = float(np.array(preds == data.y_who).sum()) / preds.shape[0]
            except Exception as e:
                acc = np.array(preds == data.y_who).sum().float() / preds.shape[0]
                print(e)
                print(preds_val, preds_batch_id)
                print(preds)
                print(target_ids_batches,index_preds_batch)
                print(preds.shape, data.y_who.shape)
        return loss.float(), acc, preds.clone().detach(), data.y_who
    
    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.learning_rate , momentum=0.9, weight_decay=2e-5)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss, acc, _, _ = self.forward(batch, mode="train")
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc, pred, y = self.forward(batch, mode="val")
        dict_report = classification_report(y.cpu().numpy(), pred.cpu().numpy(), output_dict=True, zero_division=0)
        for key in dict_report:
            if type(dict_report[key]) == dict:
                for key_small in dict_report[key]:
                    self.log('val-'+key+'-'+key_small, dict_report[key][key_small])
            else:
                self.log('val-'+key, dict_report[key])
        self.log('val_acc', acc)
        self.log('val_loss', loss)
        return dict_report
    
    def test_step(self, batch, batch_idx):
        loss, acc, pred, y = self.forward(batch, mode="test")
        dict_report = classification_report(y.cpu().numpy(), pred.cpu().numpy(), output_dict=True, zero_division=0)
        for key in dict_report:
            if type(dict_report[key]) == dict:
                for key_small in dict_report[key]:
                    self.log('test-'+ key+'-'+key_small, dict_report[key][key_small])
            else:
                self.log('test-'+key, dict_report[key])
        self.log('test_acc', acc)
        self.log('test_loss', loss)
        return dict_report

class NodeLevelMMPN(pl.LightningModule):

    def __init__(self, n_features_nodes,n_features_edge, n_features_global, message_arch, node_arch, n_embedding_group, n_output_dim_node, n_output_dim_action, 
                    second_layer=False, second_message_arch='', second_node_update_arch='',dropout_p=0.5, pooling_operation='max',
                    third_layer=False, third_message_arch='', third_node_update_arch='', loss_module=nn.MSELoss(), split_mode="", lr=0.1, limit_train=False):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = lr
        self.model = MMPN(n_features_nodes, n_features_edge, n_features_global, message_arch, node_arch, n_embedding_group, n_output_dim_node, n_output_dim_action, 
                    second_layer, second_message_arch, second_node_update_arch, 
                    third_layer, third_message_arch, third_node_update_arch, pooling_operation, dropout_p)
        
        self.loss_module = loss_module
        self.num_parameters = self.model.num_parameters

    

    def forward(self, data, mode="train"):
        x, edge_index, edge_attr, global_attr, batch_idx = data.x, data.edge_index, data.edge_attr, \
                                                            data.global_attr, \
                                                                data.batch
        num_nodes = torch.Tensor(np.array(data.num_nodes_loc)).to(torch.int64).to(device=global_attr.device).squeeze(1)
        num_edges = torch.Tensor(np.array(data.num_edges_loc)).to(torch.int64).to(device=global_attr.device).squeeze(1)
        x = self.model(x, edge_index,  edge_attr, global_attr, num_nodes, num_edges, batch_idx)

        preds_val, preds_batch_id = torch_scatter.scatter_max(x, batch_idx, dim=0)
        target_ids_batches = []
        for n in num_nodes:
            target_ids_batches.extend(np.arange(n))
        
        index_preds_batch = np.array(preds_batch_id.squeeze()).astype(int)
        preds = np.array(target_ids_batches)[index_preds_batch]
   
        target_vec = []
        for line in data.y_who_one:
            target_vec.extend(line)
        loss = self.loss_module(x.squeeze(), torch.tensor(target_vec).float())

        try:
            acc = (torch.tensor(preds) == data.y_who).sum().float() / preds.shape[0]
        except:
            try:
                # this case is a fall back in case there is only one group in the batch
                # in this case preds is a single number and not an array
                preds = np.array([preds])
                acc = (torch.tensor(preds) == data.y_who).sum().float() / preds.shape[0]
            except:
                print(preds_val, preds_batch_id)
                print(preds)
                print(target_ids_batches,index_preds_batch)
                print(preds.shape, data.y_who.shape)
        return loss.float(), acc, torch.tensor(preds), data.y_who

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.learning_rate , momentum=0.9, weight_decay=2e-5)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss, acc, _, _ = self.forward(batch, mode="train")
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc, pred, y = self.forward(batch, mode="val")
        dict_report = classification_report(y.cpu().numpy(), pred.cpu().numpy(), output_dict=True, zero_division=0)
        for key in dict_report:
            if type(dict_report[key]) == dict:
                for key_small in dict_report[key]:
                    self.log('val-'+key+'-'+key_small, dict_report[key][key_small])
            else:
                self.log('val-'+key, dict_report[key])
        self.log('val_acc', acc)
        self.log('val_loss', loss)
        return dict_report
    
    def test_step(self, batch, batch_idx):
        loss, acc, pred, y = self.forward(batch, mode="test")
        dict_report = classification_report(y.cpu().numpy(), pred.cpu().numpy(), output_dict=True, zero_division=0)
        for key in dict_report:
            if type(dict_report[key]) == dict:
                for key_small in dict_report[key]:
                    self.log('test-'+ key+'-'+key_small, dict_report[key][key_small])
            else:
                self.log('test-'+key, dict_report[key])
        self.log('test_acc', acc)
        self.log('test_loss', loss)
        return dict_report

class MMPN(nn.Module):

    def __init__(self, n_features_nodes, n_features_edges, n_features_global, message_arch, node_arch, n_embedding_group, n_output_dim_node, n_output_dim_action, 
                    second_layer=False, second_message_arch='', second_node_update_arch='', 
                    third_layer=False, third_message_arch='', third_node_update_arch='', pooling_operation='max',dropout_p=0.0):
        super(MMPN, self).__init__()
        print('Used dropout:', dropout_p)
        self.pooling_operation = pooling_operation
        self.second_layer = second_layer
        self.third_layer = third_layer
        self.message_nn = torch.nn.Sequential()
        layers = message_arch.split('-')
        n_message_dim = int(layers[-1])
        input_dim = n_features_edges+n_features_nodes+n_features_nodes+n_features_global
        for i in range(len(layers)):
            self.message_nn.add_module("Lin"+str(i), nn.Linear(input_dim, int(layers[i])))
            self.message_nn.add_module("Relu"+str(i), nn.ReLU())
            if i != len(layers) - 1:
                self.message_nn.add_module("Dropout"+str(i), nn.Dropout(dropout_p))
            input_dim = int(layers[i])
        print('Input dim message', n_features_edges+n_features_nodes+n_features_nodes+n_features_global)
    
        self.update_nn = torch.nn.Sequential()
        layers = node_arch.split('-')
        input_dim = n_message_dim + n_features_nodes + n_features_global
        n_embedding_dim = int(layers[-1])
        print(layers, n_embedding_dim)
        for i in range(len(layers)):
            self.update_nn.add_module("Lin"+str(i), nn.Linear(input_dim, int(layers[i])))
            self.update_nn.add_module("Relu"+str(i), nn.ReLU())
            if i != len(layers) - 1:
                self.update_nn.add_module("Dropout"+str(i), nn.Dropout(dropout_p))
            input_dim = int(layers[i])

        # update global
        self.to_global_nn = nn.Linear(n_embedding_dim+n_features_global, n_embedding_group)

        print('Input dim update: ', n_message_dim + n_features_nodes + n_features_global)


        if second_layer:
            # second round of GNN
            self.message_nn_2 = torch.nn.Sequential()
            layers = second_message_arch.split('-')
            n_message_dim_2 = int(layers[-1])
            print(layers, n_message_dim_2)
            print("Second message dim", n_message_dim_2)
            input_dim = n_message_dim+n_embedding_dim+n_embedding_dim+n_embedding_group
            for i in range(len(layers)):
                self.message_nn_2.add_module("Lin"+str(i), nn.Linear(input_dim, int(layers[i])))
                self.message_nn_2.add_module("Relu"+str(i), nn.ReLU())
                if i != len(layers) - 1:
                    self.message_nn_2.add_module("Dropout"+str(i), nn.Dropout(dropout_p))
                input_dim = int(layers[i])


            self.update_nn_2 = torch.nn.Sequential()
            layers = second_node_update_arch.split('-')
            
            print('Input to second update', n_message_dim_2, n_embedding_dim, n_embedding_group)
            input_dim = n_message_dim_2+n_embedding_dim + n_embedding_group
            n_embedding_dim = int(layers[-1])
            for i in range(len(layers)):
                self.update_nn_2.add_module("Lin"+str(i), nn.Linear(input_dim, int(layers[i])))
                self.update_nn_2.add_module("Relu"+str(i), nn.ReLU())
                if i != len(layers) - 1:
                    self.update_nn_2.add_module("Dropout"+str(i), nn.Dropout(dropout_p))
                input_dim = int(layers[i])
            
            self.to_global_nn_2 = nn.Linear(n_embedding_dim+n_embedding_group, n_embedding_group)

        self.embedding_nn = nn.Linear(n_embedding_dim,n_output_dim_node)
  

        self.last_relu = nn.ReLU()

        self.num_parameters = self.count_parameters(self )
        self.print_parameters()
    
    def print_parameters(self):
        print('Message model:', self.count_parameters(self.message_nn ), self.message_nn)
        print('Update model:', self.count_parameters(self.update_nn ), self.update_nn)
        if self.second_layer:
            print('Message model 2:', self.count_parameters(self.message_nn_2 ), self.message_nn_2)
            print('Update model 2:', self.count_parameters(self.update_nn_2 ), self.update_nn_2)
        print('Last layer', self.count_parameters(self.embedding_nn ), self.embedding_nn)
        print('Overall:', self.count_parameters(self ))

    def count_parameters(self, model): 
        return sum(p.numel() for p in model.parameters() if p.requires_grad) 

    ## TODO: change name once done
    def forward(self, nodes, edge_indices, edge_attr, global_attr, num_nodes, num_edges, batch_indices, return_embeddings=False):

    
        edges_list_index_to_list = []
        overall_count = 0

        for elem in edge_indices:
            count = overall_count
            max_elem = 0
            list_to_append = []
            for edge in elem:
                max_elem = max(max_elem, max(edge))
                list_to_append.append([x+count for x in edge])
            
            if overall_count+max_elem < nodes.shape[0]:
                edges_list_index_to_list.extend(list_to_append)     
            overall_count += max_elem+1
        
        src_index = torch.tensor([x[0] for x in edges_list_index_to_list], device=nodes.device).to(torch.int64)
        target_index = torch.tensor([x[1] for x in edges_list_index_to_list], device=nodes.device).to(torch.int64)

        x_nodes = nodes
        tmp_src = torch.index_select(x_nodes, 0, src_index) # for a in x_nodes ])
        
        tmp_target = torch.index_select(x_nodes, 0, target_index) #for a in x_nodes ])
        
        # repeat global graph attribute to be part of each edge

        tmp_glob = torch.repeat_interleave(global_attr, num_edges, 0)
        
        # concatenate node and edge information
        tmp_concat = torch.cat([tmp_src, edge_attr, tmp_target, tmp_glob], 1)
        
        # first pass the message
        # input: node_source + each outgoing edge
        message = F.relu(self.message_nn(tmp_concat))
        
        
        #expand target index to be applicable for every dimension of the embedding
        
        target_index_expand = target_index.expand((message.shape[1], target_index.size(0))).T
        
        # output: message_dim for each edge+node combination
        # aggregate information through max, min, mean for the message-incoming nodes
    
        if self.pooling_operation == 'mean':
            output_tensor_aggr =  torch_scatter.scatter_mean(message, target_index_expand, dim=0) #for a in torch.Tensor(message) ])
        elif self.pooling_operation == 'min':
            output_tensor_aggr, _ =  torch_scatter.scatter_min(message, target_index_expand, dim=0)
        else:
            output_tensor_aggr, _ =  torch_scatter.scatter_max(message, target_index_expand, dim=0)
        
        tmp_glob_for_nodes = torch.repeat_interleave(global_attr, num_nodes, 0)
        
        # use aggregated message information and node features to update the node
        tmp_concat_node_update = torch.cat([x_nodes, output_tensor_aggr, tmp_glob_for_nodes], 1)
        
        updated_node_embedding = self.update_nn(tmp_concat_node_update) #.view(-1, tmp_concat_node_update.shape[2]))
        # use embedding to aggregate information for global graph attribute
        
        if self.pooling_operation == 'mean':
            aggregate_nodes = torch_scatter.scatter_mean(updated_node_embedding, batch_indices, dim=0)
        elif self.pooling_operation == 'min':
            aggregate_nodes, _ = torch_scatter.scatter_min(updated_node_embedding, batch_indices, dim=0)
        else:
            aggregate_nodes, _ = torch_scatter.scatter_max(updated_node_embedding, batch_indices, dim=0)
        
        group_embedding = F.relu(self.to_global_nn(torch.cat([aggregate_nodes, global_attr], 1)))
        
        if self.second_layer:
            tmp_src_2 = torch.index_select(updated_node_embedding, 0, src_index) # for a in x_nodes ])
            
            tmp_target_2 = torch.index_select(updated_node_embedding, 0, target_index) #for a in x_nodes ])


            tmp_glob_2 = torch.repeat_interleave(group_embedding, num_edges, 0)

            #print('Shapes for message:', tmp_src_2.shape, message.shape, tmp_target_2.shape, tmp_glob_2.shape)
            tmp_concat_2 = torch.cat([tmp_src_2, message, tmp_target_2, tmp_glob_2], 1)
            

            # first pass the message
            # input: node_source + each outgoing edge
            # reduce edge dimension
    
            message_tmp_2 = self.message_nn_2(tmp_concat_2)
            
            # resort to edges to batches
            message_2 = message_tmp_2 #.view(-1, len(src_index), message_tmp.shape[1])
            
            #expand target index to be applicable for every dimension of the embedding
            
            target_index_expand_2 = target_index.expand((message_2.shape[1], target_index.size(0))).T
            
            # output: message_dim for each edge+node combination
            # aggregate information through max (most important information) for the message-incoming nodes
        
            if self.pooling_operation == 'mean':
                output_tensor_aggr_tmp_2 =  torch_scatter.scatter_mean(message_2, target_index_expand_2, dim=0) #for a in torch.Tensor(message) ])
            elif self.pooling_operation == 'min':
                output_tensor_aggr_tmp_2, _ =  torch_scatter.scatter_min(message_2, target_index_expand_2, dim=0) #for a in torch.Tensor(message) ])
            else:
                output_tensor_aggr_tmp_2, _ =  torch_scatter.scatter_max(message_2, target_index_expand_2, dim=0) #for a in torch.Tensor(message) ])
            
            output_tensor_aggr_2 = output_tensor_aggr_tmp_2 
            
            tmp_glob_for_nodes_2 = torch.repeat_interleave(group_embedding, num_nodes, 0)
            
            # use aggregated message information and node features to update the node
            tmp_concat_node_update_2 = torch.cat([updated_node_embedding, output_tensor_aggr_2, tmp_glob_for_nodes_2], 1)
            
            updated_node_embedding = self.update_nn_2(tmp_concat_node_update_2)


        result_per_node = self.embedding_nn(updated_node_embedding)
    
        if return_embeddings:
            return torch_scatter.scatter_softmax(result_per_node, batch_indices, dim=0), updated_node_embedding
        else:
            return torch_scatter.scatter_softmax(result_per_node, batch_indices, dim=0) #, results_actions #.view(-1, x_nodes.shape[1])
 
 
class BaselineLinearType(pl.LightningModule):
    def __init__(self, architecture, n_features, n_output_dim_node, n_output_dim_action, 
                    loss_module=nn.MSELoss(), split_mode="", lr=0.1, limit_train=False, dropout_p=0.5):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = lr
        
        self.model = torch.nn.Sequential()
        layers = architecture.split('-')
    
        input_dim = n_features
        for i in range(len(layers)):
            self.model.add_module("Lin"+str(i), nn.Linear(input_dim, int(layers[i])))
            self.model.add_module("Relu"+str(i), nn.ReLU())
            self.model.add_module("Dropout"+str(i), nn.Dropout(dropout_p))
            input_dim = int(layers[i])
        self.model.add_module("Lin"+str(len(layers)), nn.Linear(input_dim, int(n_output_dim_action)))
        self.softmax = nn.Softmax(dim=1)
        self.loss_module = loss_module
        self.num_parameters = self.count_parameters(self )
        self.print_parameters()

    def print_parameters(self):
        print('Model:', self.count_parameters(self.model ), self.model)
        print('Overall:', self.count_parameters(self )) 

    def count_parameters(self, model): 
        return sum(p.numel() for p in model.parameters() if p.requires_grad) 

    def forward(self, data, mode="train"):
        x, batch_idx = data.x, data.batch
        
        num_nodes = torch.Tensor(np.array(data.num_nodes_loc)).to(torch.int64).to(device=x.device).squeeze(1)
        
        y_out = self.softmax(self.model(x))
        preds = y_out.argmax(dim=1)
        
        loss = self.loss_module(y_out, torch.nn.functional.one_hot(data.y_what.clone().detach(), num_classes=4).float()) #data.y)
        
        acc = (preds == data.y_what).sum().float() / preds.shape[0]
        return loss, acc, preds, data.y_what


    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.learning_rate , momentum=0.9, weight_decay=2e-5)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss, acc, _, _ = self.forward(batch, mode="train")
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc, pred, y = self.forward(batch, mode="val")
        dict_report = classification_report(y.cpu().numpy(), pred.cpu().numpy(), output_dict=True, zero_division=0)
        for key in dict_report:
            if type(dict_report[key]) == dict:
                for key_small in dict_report[key]:
                    self.log('val-'+key+'-'+key_small, dict_report[key][key_small])
            else:
                self.log('val-'+key, dict_report[key])
        self.log('val_acc', acc)
        self.log('val_loss', loss)
        return dict_report
    
    def test_step(self, batch, batch_idx):
        loss, acc, pred, y = self.forward(batch, mode="test")
        dict_report = classification_report(y.cpu().numpy(), pred.cpu().numpy(), output_dict=True, zero_division=0)
        for key in dict_report:
            if type(dict_report[key]) == dict:
                for key_small in dict_report[key]:
                    self.log('test-'+ key+'-'+key_small, dict_report[key][key_small])
            else:
                self.log('test-'+key, dict_report[key])
        self.log('test_acc', acc)
        self.log('test_loss', loss)
        return dict_report
        

class GlobalLevelMMPN(NodeLevelMMPN):
    def __init__(self, n_features_nodes, n_features_edge, n_features_global, message_arch, node_arch, n_embedding_group, n_output_dim_node, n_output_dim_action, 
                    second_layer=False, second_message_arch='', second_node_update_arch='', 
                    third_layer=False, third_message_arch='', third_node_update_arch='', loss_module=nn.MSELoss(), split_mode="", lr=0.1, dropout_p=0.5,
                    pooling_operation='max', limit_train=False):
        super().__init__(n_features_nodes,n_features_edge, n_features_global, message_arch, node_arch, n_embedding_group, n_output_dim_node, n_output_dim_action, 
                    second_layer, second_message_arch, second_node_update_arch,dropout_p, pooling_operation,
                    third_layer, third_message_arch, third_node_update_arch, loss_module, split_mode, lr, limit_train)
        self.model = MMPNType(n_features_nodes, n_features_edge, n_features_global, message_arch, node_arch, n_embedding_group, n_output_dim_node, n_output_dim_action, 
                    second_layer, second_message_arch, second_node_update_arch, 
                    third_layer, third_message_arch, third_node_update_arch)

    def forward(self, batch, mode="train"):
        
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        global_attr = batch.global_attr
        num_nodes = torch.Tensor(np.array(batch.num_nodes_loc)).to(torch.int64).to(device=global_attr.device).squeeze(1)
        num_edges = torch.Tensor(np.array(batch.num_edges_loc)).to(torch.int64).to(device=global_attr.device).squeeze(1)
        batch_idx=batch.batch
        actions_target = batch.y_who
        actions_target_batch_wide = torch.clone(actions_target).cpu()
        elements_unique, counts = np.unique(batch_idx, return_counts=True)
        for i in range(1, actions_target.shape[0]):
        
            actions_target_batch_wide[i] = actions_target_batch_wide[i] + np.sum(counts[:i])
        
        y = self.model(x, edge_index, edge_attr, global_attr, num_nodes, num_edges, batch_idx, actions_target_batch_wide)
        
        preds = y.argmax(dim=1)
        try: 
            one_hot = torch.nn.functional.one_hot(batch.y_what.clone().detach(), num_classes=4).float()
        except:
            print(batch.y_what)
        loss = self.loss_module(y, one_hot) #data.y)
    
        acc = (preds == batch.y_what).sum().float() / preds.shape[0]
        return loss, acc, preds, batch.y_what

 

      
        
   



class MMPNType(nn.Module):

    def __init__(self, n_features_nodes, n_features_edges, n_features_global, message_arch, node_arch, n_embedding_group, n_output_dim_node, n_output_dim_action, 
                    second_layer=False, second_message_arch='', second_node_update_arch='', 
                    third_layer=False, third_message_arch='', third_node_update_arch='', dropout_p=0.0, pooling_op='max'):
        super(MMPNType, self).__init__()
        self.second_layer = second_layer
        self.third_layer = third_layer
        self.pooling_operation = pooling_op
        self.message_nn = torch.nn.Sequential()
        layers = message_arch.split('-')
        n_message_dim = int(layers[-1])
        print(n_features_edges, n_features_nodes, n_features_global)
        input_dim = n_features_edges+n_features_nodes+n_features_nodes+n_features_global
        for i in range(len(layers)):
            self.message_nn.add_module("Lin"+str(i), nn.Linear(input_dim, int(layers[i])))
            self.message_nn.add_module("Relu"+str(i), nn.ReLU())
            if i != len(layers) - 1:
                self.message_nn.add_module("Dropout"+str(i), nn.Dropout(dropout_p))
            input_dim = int(layers[i])
        
        print('Input dim message', n_features_edges+n_features_nodes+n_features_nodes+n_features_global)
        
        self.update_nn = torch.nn.Sequential()
        layers = node_arch.split('-')
        input_dim = n_message_dim + n_features_nodes + n_features_global
        n_embedding_dim = int(layers[-1])
        print(layers, n_embedding_dim)
        for i in range(len(layers)):
            self.update_nn.add_module("Lin"+str(i), nn.Linear(input_dim, int(layers[i])))
            self.update_nn.add_module("Relu"+str(i), nn.ReLU())
            if i != len(layers) - 1:
                self.update_nn.add_module("Dropout"+str(i), nn.Dropout(dropout_p))
            input_dim = int(layers[i])

        # update global
        self.to_global_nn = nn.Linear(n_embedding_dim+n_features_global, n_embedding_group)

        
        if second_layer:
            # second round of GNN
            self.message_nn_2 = torch.nn.Sequential()
            layers = second_message_arch.split('-')
            n_message_dim_2 = int(layers[-1])
            print(layers, n_message_dim_2)
            print("Second message dim", n_message_dim_2)
            input_dim = n_message_dim+n_embedding_dim+n_embedding_dim+n_embedding_group
            for i in range(len(layers)):
                self.message_nn_2.add_module("Lin"+str(i), nn.Linear(input_dim, int(layers[i])))
                self.message_nn_2.add_module("Relu"+str(i), nn.ReLU())
                if i != len(layers) - 1:
                    self.message_nn_2.add_module("Dropout"+str(i), nn.Dropout(dropout_p))
                input_dim = int(layers[i])


            self.update_nn_2 = torch.nn.Sequential()
            layers = second_node_update_arch.split('-')
            
            print('Input to second update', n_message_dim_2, n_embedding_dim, n_embedding_group)
            input_dim = n_message_dim_2+n_embedding_dim + n_embedding_group
            n_embedding_dim = int(layers[-1])
            for i in range(len(layers)):
                self.update_nn_2.add_module("Lin"+str(i), nn.Linear(input_dim, int(layers[i])))
                self.update_nn_2.add_module("Relu"+str(i), nn.ReLU())
                if i != len(layers) - 1:
                    self.update_nn_2.add_module("Dropout"+str(i), nn.Dropout(dropout_p))
                input_dim = int(layers[i])
            
            self.to_global_nn_2 = nn.Linear(n_embedding_dim+n_embedding_group, n_embedding_group)


        self.action_prediction = nn.Linear(n_embedding_dim+n_embedding_group, n_output_dim_action)
        self.last_soft = nn.Softmax(dim=1)
        
        self.print_parameters()
    
    def print_parameters(self):
        print('Message model:', self.count_parameters(self.message_nn ), self.message_nn)
        print('Update model:', self.count_parameters(self.update_nn ), self.update_nn)
        print("Global model", self.count_parameters(self.to_global_nn ), self.to_global_nn)
        if self.second_layer:
            print('Message model 2:', self.count_parameters(self.message_nn_2 ), self.message_nn_2)
            print('Update model 2:', self.count_parameters(self.update_nn_2 ), self.update_nn_2)
            print("Global model 2", self.count_parameters(self.to_global_nn_2 ), self.to_global_nn_2)
        
        print('Last layer', self.count_parameters(self.action_prediction ), self.action_prediction)
        print('Overall:', self.count_parameters(self ))

    def count_parameters(self, model): 
        return sum(p.numel() for p in model.parameters() if p.requires_grad) 

    def forward(self, nodes, edge_indices, edge_attr, global_attr, num_nodes, num_edges, batch_indices, target_batch_wide, return_embeddings=False):
        # TYPE PREDICTION!!!!
        # dims: batch_size, num_nodes, features
        # dims: batch_size, num_nodes, features
        edges_list_index_to_list = []
        overall_count = 0
        for elem in edge_indices:
            count = overall_count
            max_elem = 0
            list_to_append = []
            for edge in elem:
                max_elem = max(max_elem, max(edge))
                list_to_append.append([x+count for x in edge])
            
            if overall_count+max_elem < nodes.shape[0]:
                edges_list_index_to_list.extend(list_to_append)     
            overall_count += max_elem+1
        
        src_index = torch.tensor([x[0] for x in edges_list_index_to_list], device=nodes.device).to(torch.int64)
        target_index = torch.tensor([x[1] for x in edges_list_index_to_list], device=nodes.device).to(torch.int64)

        x_nodes = nodes
        
        tmp_src = torch.index_select(x_nodes, 0, src_index) # for a in x_nodes ])
        
        tmp_target = torch.index_select(x_nodes, 0, target_index) #for a in x_nodes ])
        
        # repeat global graph attribute to be part of each edge
        tmp_glob = torch.repeat_interleave(global_attr, num_edges, 0)

        tmp_concat = torch.cat([tmp_src, edge_attr, tmp_target, tmp_glob], 1)
        
        # first pass the message
        # input: node_source + each outgoing edge

        message = F.relu(self.message_nn(tmp_concat))
        
        
        #expand target index to be applicable for every dimension of the embedding
        
        target_index_expand = target_index.expand((message.shape[1], target_index.size(0))).T
        
        if self.pooling_operation == 'mean':
            output_tensor_aggr =  torch_scatter.scatter_mean(message, target_index_expand, dim=0) #for a in torch.Tensor(message) ])
        elif self.pooling_operation == 'min':
            output_tensor_aggr, _ =  torch_scatter.scatter_min(message, target_index_expand, dim=0)
        else:
            output_tensor_aggr, _ =  torch_scatter.scatter_max(message, target_index_expand, dim=0)
       
        tmp_glob_for_nodes = torch.repeat_interleave(global_attr, num_nodes, 0)
        
        # use aggregated message information and node features to update the node
        tmp_concat_node_update = torch.cat([x_nodes, output_tensor_aggr, tmp_glob_for_nodes], 1)
        
        updated_node_embedding = self.update_nn(tmp_concat_node_update) #.view(-1, tmp_concat_node_update.shape[2]))
        # use embedding to aggregate information for global graph attribute
        if self.pooling_operation == 'mean':
            aggregate_nodes =  torch_scatter.scatter_mean(updated_node_embedding, batch_indices, dim=0) #for a in torch.Tensor(message) ])
        elif self.pooling_operation == 'min':
            output_teaggregate_nodesnsor_aggr, _ =  torch_scatter.scatter_min(updated_node_embedding, batch_indices, dim=0)
        else:
            aggregate_nodes, _ =  torch_scatter.scatter_max(updated_node_embedding, batch_indices, dim=0)
        
        group_embedding = F.relu(self.to_global_nn(torch.cat([aggregate_nodes, global_attr], 1)))
        
        if self.second_layer:
            tmp_src_2 = torch.index_select(updated_node_embedding, 0, src_index) # for a in x_nodes ])
            
            tmp_target_2 = torch.index_select(updated_node_embedding, 0, target_index) #for a in x_nodes ])


            tmp_glob_2 = torch.repeat_interleave(group_embedding, num_edges, 0)

            
            tmp_concat_2 = torch.cat([tmp_src_2, message, tmp_target_2, tmp_glob_2], 1)
            

            # first pass the message
            # input: node_source + each outgoing edge
            # reduce edge dimension
            
            message_tmp_2 = self.message_nn_2(tmp_concat_2)
            
            # resort to edges to batches
            message_2 = message_tmp_2 #.view(-1, len(src_index), message_tmp.shape[1])
            
            #expand target index to be applicable for every dimension of the embedding
            
            target_index_expand_2 = target_index.expand((message_2.shape[1], target_index.size(0))).T
            
            # output: message_dim for each edge+node combination
            # aggregate information through max (most important information) for the message-incoming nodes
            if self.pooling_operation == 'mean':
                output_tensor_aggr_2 =  torch_scatter.scatter_mean(message_2, target_index_expand_2, dim=0) #for a in torch.Tensor(message) ])
            elif self.pooling_operation == 'min':
                output_tensor_aggr_2, _ =  torch_scatter.scatter_min(message_2, target_index_expand_2, dim=0)
            else:
                output_tensor_aggr_2, _ =  torch_scatter.scatter_max(message_2, target_index_expand_2, dim=0)
            
            tmp_glob_for_nodes_2 = torch.repeat_interleave(group_embedding, num_nodes, 0)
            
            # use aggregated message information and node features to update the node
            tmp_concat_node_update_2 = torch.cat([updated_node_embedding, output_tensor_aggr_2, tmp_glob_for_nodes_2], 1)
            
            updated_node_embedding = self.update_nn_2(tmp_concat_node_update_2)
            if self.pooling_operation == 'mean':
                aggregate_nodes =  torch_scatter.scatter_mean(updated_node_embedding, batch_indices, dim=0) #for a in torch.Tensor(message) ])
            elif self.pooling_operation == 'min':
                output_teaggregate_nodesnsor_aggr, _ =  torch_scatter.scatter_min(updated_node_embedding, batch_indices, dim=0)
            else:
                aggregate_nodes, _ =  torch_scatter.scatter_max(updated_node_embedding, batch_indices, dim=0)
            
            group_embedding = F.relu(self.to_global_nn_2(torch.cat([aggregate_nodes, group_embedding], 1)))
        
        chosen_nodes = torch.index_select(updated_node_embedding, 0, target_batch_wide)
        #print(updated_node_embedding, chosen_nodes)
        tmp_concat_action_pred = torch.cat([chosen_nodes, group_embedding], 1)

        
        results_actions =  self.last_soft(self.action_prediction(tmp_concat_action_pred))
        if return_embeddings:
            return results_actions, tmp_concat_action_pred
        else:
            return results_actions