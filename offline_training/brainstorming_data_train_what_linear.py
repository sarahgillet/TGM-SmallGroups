#!/usr/bin/env python
# coding: utf-8

# In[137]:


import utils
import config
import wandb
import torch
from dataset import SummerschoolDataset, BrainstormingDataset, BrainstormingDatasetFixedSize
import MMPN
import torch.nn as nn

import absl.app
import absl.flags
from sklearn import preprocessing
import pickle

import numpy as np


utils.set_random_seed(42)

FLAGS_DEF = utils.define_flags_with_default(
    device= 'cpu',
    h_lookback= 10,
    use_samples=1,
    h_TEST_SIZE= 0.1,
    h_VALID_SIZE= 1.0/6.0,
    h_BATCH_SIZE = 32,
    h_SEED = 42,
    early_stopping=False,
    tune_learning_rate=False,
    n_epochs=500,
    group_embedding_dim=6, # was 12 for all other runs
    project_config_name='BrainstormingExploreOriginal23Linear',
    google_sheet_id=1, # give ID of sheet starting with 1
    wandb_project_name="Brainstorming_Explore_Original_2_3_linear",
    use_pretrained=False,
    pretrained_path="",
    arch='16-32-8',
    loss_module='MSELoss',
    limit_train=False,
    pooling_operation='min',
    dropout=0.5,
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


    # In[164]:

    print("Starting training")
    dataset_org=BrainstormingDatasetFixedSize("../training_data/", "annotated_data_what_all/", aggr_func = 'min', type='what')
    utils.set_random_seed(FLAGS.h_SEED)
    train_dataset, test_dataset, valid_dataset, split_type = utils.generateTrainTestValidDataset(dataset_org, FLAGS.h_SEED, FLAGS.h_TEST_SIZE, FLAGS.h_VALID_SIZE, mixing_episodes=False)
    # normalize training data only
    #print(train_dataset.data.x)
    print("Length of y_who:", len(test_dataset.data.y_who), "Length of x: ", len(test_dataset.data.x))
    print(train_dataset.data.x)
    scaler_n = preprocessing.StandardScaler().fit(train_dataset.data.x)
    train_dataset.transform_scaler = scaler_n
    train_dataset.activate_transform()
    valid_dataset.transform_scaler = scaler_n
    valid_dataset.activate_transform()
    test_dataset.transform_scaler = scaler_n
    loss_module = nn.MSELoss()
    if FLAGS.loss_module == "CrossEntropyLoss":
        loss_module = nn.CrossEntropyLoss()
    graph_train_loader, graph_val_loader, graph_test_loader = utils.loadData(train_dataset, valid_dataset, test_dataset, FLAGS.h_BATCH_SIZE)
    model, result, val_result, test_result, run_id, num_params, path = utils.train_graph_classifier(model_name="Brainstorming-Linear-What", wandb_project_name=FLAGS.wandb_project_name, 
                                        h_SEED=FLAGS.h_SEED, device=FLAGS.device,
                                        graph_train_loader=graph_train_loader, graph_val_loader=graph_val_loader,
                                        graph_test_loader=graph_test_loader,
                                        early_stopping=FLAGS.early_stopping,
                                        return_path=True,
                                        tune_lr=FLAGS.tune_learning_rate,
                                        max_epochs=FLAGS.n_epochs,
                                        loss_module=loss_module,
                                        dropout_p = FLAGS.dropout,
                                        architecture=FLAGS.arch,
                                        n_output_dim_node = 2, 
                                        n_output_dim_action = 4, 
                                        n_features = int(config.NUM_NODE_FEATURES)*2, 
                                        split_mode = split_type, 
                                        lr=0.1)
    dict_double_val = utils.getDoubleDict(val_result)
    dict_double_test = utils.getDoubleDict(test_result)
    NAME_SCALER_FILE = "../scalers/"+FLAGS.project_config_name+"-"+split_type+"-"+run_id+"_fixed.pkl"
    #pickle.dump(scaler_n, open(NAME_SCALER_FILE, 'wb'))
    utils.writeToGoogleSheetsLinear(FLAGS.project_config_name, FLAGS.google_sheet_id-1, run_id, 
                FLAGS.arch, split_type, FLAGS.h_SEED, dict_double_val, dict_double_test, 
                val_result[0]['val_acc'], test_result[0]['test_acc'], num_params,
                FLAGS.h_BATCH_SIZE, FLAGS.loss_module, len(train_dataset.data.y_what),
                FLAGS.limit_train if not FLAGS.limit_train else 'Groups of '+str(FLAGS.group_size_training)+' , #Ep: '+str(FLAGS.limit_amount_of_train_val),
                FLAGS.pooling_operation, FLAGS.dropout, use_drive=FLAGS.use_drive)                                      
    wandb.finish()
    if not FLAGS.do_test:
        return
    dataset_org=BrainstormingDatasetFixedSize("../training_data/raw", "data_aggregated_four_what/", test_data=True, group_sizes_train="4", type='what', aggr_func='min')
    utils.set_random_seed(FLAGS.h_SEED)
    dataset_org.transform_scaler = scaler_n
    dataset_org.activate_transform()
    split_type="None"

    graph_train_loader, graph_val_loader, graph_test_loader = utils.loadData(dataset_org, dataset_org, dataset_org, FLAGS.h_BATCH_SIZE)
    print("Loading from", path)
    model, result, val_result, test_result, run_id, num_params = utils.train_graph_classifier(model_name="Brainstorming-Linear-What", wandb_project_name=FLAGS.wandb_project_name, 
                                        h_SEED=FLAGS.h_SEED, device=FLAGS.device,
                                        graph_train_loader=graph_train_loader, graph_val_loader=graph_val_loader,
                                        graph_test_loader=graph_test_loader,
                                        early_stopping=FLAGS.early_stopping,
                                        tune_lr=FLAGS.tune_learning_rate,
                                        load_pretrained=True,
                                        pretrained_filename=path,
                                        max_epochs=FLAGS.n_epochs,
                                        loss_module=loss_module,
                                        dropout_p = FLAGS.dropout,
                                        architecture=FLAGS.arch,
                                        n_output_dim_node = 2, 
                                        n_output_dim_action = 4, 
                                        n_features = int(config.NUM_NODE_FEATURES)*2, 
                                        split_mode = split_type, 
                                        lr=0.1)
    dict_double_val = utils.getDoubleDict(val_result)
    print(val_result)
    #print(FLAGS.project_config_name,)
    utils.writeToGoogleSheetsLinear(FLAGS.project_config_name, FLAGS.google_sheet_id, run_id, 
                FLAGS.arch, split_type, FLAGS.h_SEED, dict_double_val, dict_double_val, 
                val_result[0]['val_acc'], test_result[0]['test_acc'], num_params,
                FLAGS.h_BATCH_SIZE, FLAGS.loss_module, len(train_dataset.data.y_who),
                FLAGS.limit_train if not FLAGS.limit_train else 'Groups of '+str(FLAGS.group_size_training)+' , #Ep: '+str(FLAGS.limit_amount_of_train_val),
                FLAGS.pooling_operation, FLAGS.dropout, use_drive=FLAGS.use_drive)      

if __name__ == '__main__':
    absl.app.run(main)
