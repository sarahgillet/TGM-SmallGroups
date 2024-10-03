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
import os
import pickle


utils.set_random_seed(42)

FLAGS_DEF = utils.define_flags_with_default(
    device= 'cpu',
    h_lookback= 10,
    use_samples=1,
    h_TEST_SIZE= 1,
    h_VALID_SIZE= 0,
    h_BATCH_SIZE = 30,
    h_SEED = 42,
    memory_type='LSTM',
    early_stopping=False,
    tune_learning_rate=False,
    n_epochs=500,
    group_embedding_dim=6, # was 12 for all other runs
    project_config_name='BrainstormingExploreOriginal4',
    google_sheet_id=1, # give ID of sheet starting with 1
    wandb_project_name="Brainstorming_Explore_Original_4",
    GNN_second_layer=False,
    GNN_third_layer=False,
    h_mess_arch_1 = '4',
    h_node_arch_1 = '2',
    h_mess_arch_2 = '4',
    h_node_arch_2 = '2',    
    dropout=0.0,
    use_pretrained=True,
    #pretrained_path="/home/rpl/GNNImitationLearning/GNNGroupImitationLearning/checkpoints/GraphLevelBrainstorming-MMPN-Who/cc77dd8b6a4c4667bca5f98269bf9bc3/epoch=292-val_loss=0.23915-val_acc=0.56170.ckpt"
    #pretrained_path="/home/rpl/GNNImitationLearning/GNNGroupImitationLearning/checkpoints/GraphLevelBrainstorming-MMPN-Who/1a4e8abb67f642449cef349e03ac8ba4/epoch=0-val_loss=0.23916-val_acc=0.59838.ckpt"
    pretrained_path="/home/rpl/GNNImitationLearning/GNNGroupImitationLearning/checkpoints/GraphLevelBrainstorming-MMPN-Who/3fdc611410be4e4080d6f62ca9246caf/epoch=71-val_loss=0.01687-val_acc=0.97800.ckpt",
    #pretrained_path="/home/rpl/GNNImitationLearning/GNNGroupImitationLearning/checkpoints/GraphLevelBrainstorming-MMPN-Who/c31b54672b1446f180d5537712323fc6/epoch=258-val_loss=0.23915-val_acc=0.53249.ckpt"
    loss_module='MSELoss',
    group_size_training='3',
    limit_train=False,
    limit_amount_of_train_val=12,
    limit_amount_test = 6,
    pooling_operation='min',
    do_test=False
)







def main(argv):


    FLAGS = absl.flags.FLAGS

    variant = utils.get_user_flags(FLAGS, FLAGS_DEF)
    print(variant)
    # Hyperparameters


    # Prepare the dataset
    utils.set_random_seed(FLAGS.h_SEED)
    # mixing_episodes is the parameter that determines if we split train test valid data by episodes or in the whole dataset
    
    # find scaler
    folder = '/'.join(FLAGS.pretrained_path.split('/')[:-1])
    # find a pkl file in the folder

    for file in os.listdir(folder):
        # find file with ending .pkl
        if file.endswith('.pkl'):
            print(file)
            # read the file
            scalers = pickle.load(open(folder+'/'+file, 'rb'))
    # In[164]:
    # testing
    print("Starting testing")
    dataset_org=BrainstormingDataset("../training_data/raw", "data_aggregated_four/", test_data=True, group_sizes_train="4")
    utils.set_random_seed(FLAGS.h_SEED)
    dataset_org.transform_scaler = scalers
    dataset_org.activate_transform()
    split_type="None"

    graph_train_loader, graph_val_loader, graph_test_loader = utils.loadData(dataset_org, dataset_org, dataset_org, FLAGS.h_BATCH_SIZE)
    print("Loading from", FLAGS.pretrained_path)
    model, result, val_result, test_result, run_id, num_params = utils.train_graph_classifier(model_name="Brainstorming-MMPN-Who", wandb_project_name=FLAGS.wandb_project_name, 
                                        h_SEED=FLAGS.h_SEED, device=FLAGS.device,
                                        graph_train_loader=graph_test_loader, graph_val_loader=graph_test_loader,graph_test_loader=graph_test_loader,
                                        early_stopping=FLAGS.early_stopping,
                                        tune_lr=FLAGS.tune_learning_rate,
                                        max_epochs=FLAGS.n_epochs,
                                        load_pretrained=True,
                                        pretrained_filename=FLAGS.pretrained_path,
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
    print(val_result)
    #print(FLAGS.project_config_name,)
    utils.writeToGoogleSheetsSimple(FLAGS.project_config_name, FLAGS.google_sheet_id-1, run_id, 
                        FLAGS.h_mess_arch_1, FLAGS.h_node_arch_1, FLAGS.h_mess_arch_2, FLAGS.h_node_arch_2, 
                        split_type, FLAGS.h_SEED, dict_double_val, dict_double_val, 
                        val_result[0]['val_acc'], test_result[0]['test_acc'], num_params, 
                        FLAGS.h_BATCH_SIZE, FLAGS.loss_module, len(dataset_org.data.y_who),
                        FLAGS.limit_train if not FLAGS.limit_train else 'Groups of '+str(FLAGS.group_size_training)+' , #Ep: '+str(FLAGS.limit_amount_of_train_val), 
                        FLAGS.group_embedding_dim, FLAGS.pooling_operation, FLAGS.dropout)
    wandb.finish()


if __name__ == '__main__':
    absl.app.run(main)
