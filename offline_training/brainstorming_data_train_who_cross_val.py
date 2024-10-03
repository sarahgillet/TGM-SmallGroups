#!/usr/bin/env python
# coding: utf-8

# In[137]:


import utils
import config
import wandb
import torch
from dataset import SummerschoolDataset, BrainstormingDataset
import MMPN
import torch.nn as nn

import absl.app
import absl.flags
from sklearn import preprocessing
import pickle


utils.set_random_seed(42)

FLAGS_DEF = utils.define_flags_with_default(
    device= 'cpu',
    h_lookback= 10,
    use_samples=1,
    h_TEST_SIZE= 0.1,
    h_VALID_SIZE= 1.0/6.0,
    h_BATCH_SIZE = 30,
    h_SEED = 42,
    memory_type='LSTM',
    early_stopping=False,
    tune_learning_rate=False,
    n_epochs=500,
    group_embedding_dim=2, # was 6 for all other runs
    project_config_name='BrainstormingExplore_Limited_Training_Data',
    google_sheet_id=2, # give ID of sheet starting with 1
    wandb_project_name="Brainstorming_Experimental_fixed_size_alt1_v0",
    GNN_second_layer=False,
    GNN_third_layer=False,
    h_mess_arch_1 = '4',
    h_node_arch_1 = '2',
    h_mess_arch_2 = '4',
    h_node_arch_2 = '2',
    dropout=0.0,
    use_pretrained=False,
    pretrained_path="",
    loss_module='MSELoss',
    group_size_training='3',
    limit_train=False,
    limit_amount_of_train_val=12,
    limit_amount_test = 6,
    num_folds = 6,
    pooling_operation='min',
    do_test=False,
    use_drive=False
)







def main(argv):


    FLAGS = absl.flags.FLAGS

    variant = utils.get_user_flags(FLAGS, FLAGS_DEF)
    print(variant)
    # Hyperparameters


    # Prepare the dataset
    utils.set_random_seed(FLAGS.h_SEED)
    # mixing_episodes is the parameter that determines if we split train test valid data by episodes or in the whole dataset

    loss_module = nn.MSELoss()
    if FLAGS.loss_module == "CrossEntropyLoss":
        loss_module = nn.CrossEntropyLoss()
    # In[164]:

    print("Starting training")
    dataset_org=BrainstormingDataset("../training_data/", "annotated_data_brainstorming/")
    utils.set_random_seed(FLAGS.h_SEED)
    data_over_folds = {'ident':'', 'num_decisions':[],'num_params': 0,'val': {'acc':[], 'macro avg: f1':[]},'test': {'acc':[], 'macro avg: f1':[]}}
    data_over_folds_test = {'ident':'', 'num_decisions':[],'num_params': 0,'val': {'acc':[], 'macro avg: f1':[]},'test': {'acc':[], 'macro avg: f1':[]}}
    data_over_folds_test['path'] = []
    data_over_folds_test['scalers'] = []
    common_ident = ''
    if FLAGS.limit_train:
            train_valid_datasets, test_dataset, split_type = utils.generateTrainTestValidDataset(
                dataset_org, FLAGS.h_SEED, FLAGS.h_TEST_SIZE, FLAGS.h_VALID_SIZE, mixing_episodes=False, 
                fixed_group_size_training=FLAGS.group_size_training, 
                dataset_size_train_val=FLAGS.limit_amount_of_train_val, dataset_size_test=FLAGS.limit_amount_test, folds=True, num_folds=FLAGS.num_folds)
    else:
        raise NotImplementedError
    for fold in range(FLAGS.num_folds):
        train_dataset = train_valid_datasets[fold][0]
        valid_dataset = train_valid_datasets[fold][1]
            #train_dataset, test_dataset, valid_dataset, split_type = utils.generateTrainTestValidDataset(dataset_org, FLAGS.h_SEED, FLAGS.h_TEST_SIZE, FLAGS.h_VALID_SIZE, mixing_episodes=False)
        # normalize training data only
        #print(train_dataset.data.x)
        print(len(train_dataset.data.y_who))
        scaler_n = preprocessing.StandardScaler().fit(train_dataset.data.x)
        print(scaler_n.mean_, scaler_n.var_) #, scaler_e.mean_, scaler_g.mean_)
        # train_dataset.data.x = torch.Tensor(scaler_n.transform(train_dataset.data.x))
        scaler_e = preprocessing.StandardScaler().fit(train_dataset.data.edge_attr)
        # train_dataset.data.edge_attr = torch.Tensor(scaler_e.transform(train_dataset.data.edge_attr))
        scaler_g = preprocessing.StandardScaler().fit(train_dataset.data.global_attr)
        train_dataset.transform_scaler = [scaler_n, scaler_e, scaler_g]
        train_dataset.activate_transform()
        valid_dataset.transform_scaler = [scaler_n, scaler_e, scaler_g]
        valid_dataset.activate_transform()
        test_dataset.transform_scaler = [scaler_n, scaler_e, scaler_g]
        test_dataset.activate_transform()

        # train_dataset.data.global_attr = torch.Tensor(scaler_g.transform(train_dataset.data.global_attr))
        # print(scaler_n.mean_, scaler_n.var_, scaler_e.mean_, scaler_g.mean_)
        # # apply normalization to test and valid_dataset
        # test_dataset.data.x = torch.Tensor(scaler_n.transform(test_dataset.data.x))
        # test_dataset.data.edge_attr = torch.Tensor(scaler_e.transform(test_dataset.data.edge_attr))
        # test_dataset.data.global_attr = torch.Tensor(scaler_g.transform(test_dataset.data.global_attr))
        # valid_dataset.data.x = torch.Tensor(scaler_n.transform(valid_dataset.data.x))
        # valid_dataset.data.edge_attr = torch.Tensor(scaler_e.transform(valid_dataset.data.edge_attr))
        # valid_dataset.data.global_attr = torch.Tensor(scaler_g.transform(valid_dataset.data.global_attr))
        graph_train_loader, graph_val_loader, graph_test_loader = utils.loadData(train_dataset, valid_dataset, test_dataset, FLAGS.h_BATCH_SIZE)
        model, result, val_result, test_result, run_id, num_params, path = utils.train_graph_classifier(model_name="Brainstorming-MMPN-Who", wandb_project_name=FLAGS.wandb_project_name, 
                                            h_SEED=FLAGS.h_SEED, device=FLAGS.device,
                                            graph_train_loader=graph_train_loader, graph_val_loader=graph_val_loader,
                                            graph_test_loader=graph_test_loader,
                                            early_stopping=FLAGS.early_stopping,
                                            tune_lr=FLAGS.tune_learning_rate,
                                            max_epochs=FLAGS.n_epochs,
                                            return_path=True,
                                            loss_module=loss_module,
                                            n_features_nodes = int(config.NUM_NODE_FEATURES), 
                                            n_features_edge = int(config.NUM_EDGE_FEATURES),
                                            n_features_global = int(config.NUM_GLOB_FEATURES),
                                            message_arch = FLAGS.h_mess_arch_1, 
                                            node_arch = FLAGS.h_node_arch_1, 
                                            n_embedding_group = FLAGS.group_embedding_dim, 
                                            n_output_dim_node = 1, 
                                            n_output_dim_action = 1, 
                                            second_layer=FLAGS.GNN_second_layer, 
                                            second_message_arch=FLAGS.h_mess_arch_2, 
                                            second_node_update_arch=FLAGS.h_node_arch_2, 
                                            dropout_p = FLAGS.dropout,
                                            split_mode = split_type, 
                                            limit_train = FLAGS.limit_train,
                                            pooling_operation=FLAGS.pooling_operation,
                                            lr=0.1)
        dict_double_val = utils.getDoubleDict(val_result)
        dict_double_test = utils.getDoubleDict(test_result)
        NAME_SCALER_FILE = "../scalers/"+FLAGS.project_config_name+"-"+split_type+"-"+run_id+".pkl"
        #pickle.dump([scaler_n, scaler_e, scaler_g], open(NAME_SCALER_FILE, 'wb'))
        if common_ident == '':
            common_ident = run_id
            data_over_folds['ident'] = run_id
            data_over_folds['num_params'] = num_params
            data_over_folds_test['ident'] = run_id
            data_over_folds_test['num_params'] = num_params
        # utils.writeToGoogleSheetsSimpleCrossVal(FLAGS.project_config_name, FLAGS.google_sheet_id, run_id, common_ident,
        #                     FLAGS.h_mess_arch_1, FLAGS.h_node_arch_1, FLAGS.h_mess_arch_2, FLAGS.h_node_arch_2, 
        #                     split_type, FLAGS.h_SEED, dict_double_val, dict_double_test, 
        #                     val_result[0]['val_acc'], test_result[0]['test_acc'], num_params, 
        #                     FLAGS.h_BATCH_SIZE, FLAGS.loss_module, len(train_dataset.data.y_who),
        #                     FLAGS.limit_train if not FLAGS.limit_train else 'Groups of '+str(FLAGS.group_size_training)+' , #Ep: '+str(FLAGS.limit_amount_of_train_val),  FLAGS.group_embedding_dim,
        #                     FLAGS.pooling_operation, FLAGS.dropout)                                      
        data_over_folds['num_decisions'].append(len(train_dataset.data.y_who))
        data_over_folds['val']['acc'].append(val_result[0]['val_acc'])
        data_over_folds['test']['acc'].append(test_result[0]['test_acc'])
        data_over_folds['test']['macro avg: f1'].append(dict_double_test['macro avg']['f1'])
        data_over_folds['val']['macro avg: f1'].append(dict_double_val['macro avg']['f1'])
        wandb.finish()
        if FLAGS.do_test:
            data_over_folds_test['path'].append(path)
            data_over_folds_test['scalers'].append([scaler_n, scaler_e, scaler_g])       
    utils.writeToGoogleSheetsCrossValSummary(FLAGS.project_config_name, FLAGS.google_sheet_id-1, data_over_folds,
                            FLAGS.h_mess_arch_1, FLAGS.h_node_arch_1, FLAGS.h_mess_arch_2, FLAGS.h_node_arch_2, 
                            split_type, FLAGS.h_SEED, FLAGS.h_BATCH_SIZE, FLAGS.loss_module,
                            FLAGS.limit_train if not FLAGS.limit_train else 'Groups of '+str(FLAGS.group_size_training)+' , #Ep: '+str(FLAGS.limit_amount_of_train_val), FLAGS.group_embedding_dim,
                            FLAGS.pooling_operation, FLAGS.dropout,use_drive=FLAGS.use_drive)
    
    if FLAGS.do_test:
        for i in range(len(data_over_folds_test['path'])):
            path = data_over_folds_test['path'][i]
            dataset_org=BrainstormingDataset("../training_data/raw", "data_aggregated_four/", test_data=True, group_sizes_train="4")
            utils.set_random_seed(FLAGS.h_SEED)
            dataset_org.transform_scaler = data_over_folds_test['scalers'][i]
            dataset_org.activate_transform()
            split_type="None"

            graph_train_loader, graph_val_loader, graph_test_loader = utils.loadData(dataset_org, dataset_org, dataset_org, FLAGS.h_BATCH_SIZE)
            print("Loading from", path)
            model, result, val_result, test_result, run_id, num_params = utils.train_graph_classifier(model_name="Brainstorming-MMPN-Who", wandb_project_name=FLAGS.wandb_project_name, 
                                                h_SEED=FLAGS.h_SEED, device=FLAGS.device,
                                                graph_train_loader=graph_test_loader, graph_val_loader=graph_test_loader,graph_test_loader=graph_test_loader,
                                                early_stopping=FLAGS.early_stopping,
                                                tune_lr=FLAGS.tune_learning_rate,
                                                max_epochs=FLAGS.n_epochs,
                                                load_pretrained=True,
                                                pretrained_filename=path,
                                                n_features_nodes = int(config.NUM_NODE_FEATURES), 
                                                n_features_edge = int(config.NUM_EDGE_FEATURES),
                                                n_features_global = int(config.NUM_GLOB_FEATURES),
                                                message_arch = FLAGS.h_mess_arch_1, 
                                                node_arch = FLAGS.h_node_arch_1, 
                                                n_embedding_group = FLAGS.group_embedding_dim, 
                                                n_output_dim_node = 1, 
                                                n_output_dim_action = 1, 
                                                second_layer=FLAGS.GNN_second_layer, 
                                                second_message_arch=FLAGS.h_mess_arch_2, 
                                                second_node_update_arch=FLAGS.h_node_arch_2, 
                                                #split_mode = split_type, 
                                                lr=0.1)
            dict_double_val = utils.getDoubleDict(val_result)
            dict_double_test = utils.getDoubleDict(test_result)
            data_over_folds_test['num_decisions'].append(len(train_dataset.data.y_who))
            data_over_folds_test['val']['acc'].append(val_result[0]['val_acc'])
            data_over_folds_test['test']['acc'].append(test_result[0]['test_acc'])
            data_over_folds_test['test']['macro avg: f1'].append(dict_double_test['macro avg']['f1'])
            data_over_folds_test['val']['macro avg: f1'].append(dict_double_val['macro avg']['f1'])
            print(val_result)
            #print(FLAGS.project_config_name,)
            # utils.writeToGoogleSheetsSimple(FLAGS.project_config_name, FLAGS.google_sheet_id+2, run_id, 
            #                     FLAGS.h_mess_arch_1, FLAGS.h_node_arch_1, FLAGS.h_mess_arch_2, FLAGS.h_node_arch_2, 
            #                     split_type, FLAGS.h_SEED, dict_double_val, dict_double_val, 
            #                     val_result[0]['val_acc'], test_result[0]['test_acc'], num_params, 
            #                     FLAGS.h_BATCH_SIZE, FLAGS.loss_module, len(dataset_org.data.y_who),
            #                     FLAGS.limit_train if not FLAGS.limit_train else 'Groups of '+str(FLAGS.group_size_training)+' , #Ep: '+str(FLAGS.limit_amount_of_train_val), 
            #                     FLAGS.group_embedding_dim, FLAGS.pooling_operation, FLAGS.dropout)
        utils.writeToGoogleSheetsCrossValSummary(FLAGS.project_config_name, FLAGS.google_sheet_id, data_over_folds_test,
                            FLAGS.h_mess_arch_1, FLAGS.h_node_arch_1, FLAGS.h_mess_arch_2, FLAGS.h_node_arch_2, 
                            split_type, FLAGS.h_SEED, FLAGS.h_BATCH_SIZE, FLAGS.loss_module,
                            FLAGS.limit_train if not FLAGS.limit_train else 'Groups of '+str(FLAGS.group_size_training)+' , #Ep: '+str(FLAGS.limit_amount_of_train_val), FLAGS.group_embedding_dim,
                            FLAGS.pooling_operation, FLAGS.dropout, use_drive=FLAGS.use_drive)

if __name__ == '__main__':
    absl.app.run(main)
