#!/usr/bin/env python
# coding: utf-8

# In[137]:


import utils
import config
import wandb
import torch
from dataset import SummerschoolDataset
import MMPN
import torch.nn as nn

import absl.app
import absl.flags


utils.set_random_seed(42)

FLAGS_DEF = utils.define_flags_with_default(
    device= 'cpu',
    h_lookback= 10,
    use_samples=1,
    h_TEST_SIZE= 0.06,
    h_VALID_SIZE= 0.2,
    h_BATCH_SIZE = 28,
    h_SEED = 42,
    memory_type='LSTM',
    early_stopping=False,
    tune_learning_rate=False,
    n_epochs=200,
    group_embedding_dim='None', # was 12 for all other runs
    project_config_name='Teenagers-Who-AudioNorm-ExploreFineTuning-Linear',
    google_sheet_id=5, # give ID of sheet starting with 1
    wandb_project_name="GNN-Teenager-Who-SelectedFeatures-ExploreFineTuning",
    loss_module = 'MSELOSS',
    layer_1=64,
    layer_2=32,
    layer_linear='4',
    second_layer_memory=True,
    selected_indices='4;5;9;10;12',
    augment=False,
    mix_episodes = True,
    teen_group = 'WALLE',
    pretrained = True, 
    pretrained_path = '/home/rpl/GNNImitationLearning/GNNGroupImitationLearning/checkpoints/GraphLevelNodeLevelMMPN/643de42ed1d4437b8ea192e8e39826a4/epoch=881-val_loss=0.18800-val_acc=0.40371.ckpt'
)

def get_data_and_train(dataset_org, kept_indices_manual_selection, feature_sel_dict, training_dataset, FLAGS, finetune=False, path=''):
    # if training_dataset=='Full':
    #     training_dataset=None
    train_dataset, test_dataset, valid_dataset, test_dataset_gen = utils.generateTrainDoubleTestValidDataset(dataset_org, FLAGS.h_SEED, FLAGS.h_TEST_SIZE, FLAGS.h_VALID_SIZE, fixed_group=training_dataset if training_dataset!='Full' else None)
    split_type = "double_test_ep_mixed"
    
    print(len(train_dataset.data.y_what),len(test_dataset.data.y_what), len(valid_dataset.data.y_what) , len(test_dataset_gen.data.y_what) )
    train_dataset = utils.upsampleTrain(train_dataset, 10, lambda x: x.y_what )
    print(len(train_dataset.data.y_what))
    graph_train_loader, graph_val_loader, graph_test_loader = utils.loadData(train_dataset, valid_dataset, test_dataset, FLAGS.h_BATCH_SIZE)
    model, result, val_result, test_result, run_id, num_params, path  = utils.train_graph_classifier(model_name="SingleVectorNetworkType", wandb_project_name=FLAGS.wandb_project_name, 
                                        h_SEED=FLAGS.h_SEED, device=FLAGS.device,
                                        graph_train_loader=graph_train_loader, graph_val_loader=graph_val_loader,
                                        graph_test_loader=graph_test_loader,
                                        early_stopping=FLAGS.early_stopping,
                                        tune_lr=FLAGS.tune_learning_rate,
                                        return_path = True,
                                        load_pretrained=finetune,
                                        pretrained_filename=path,
                                        train_pretrained=finetune,
                                        max_epochs=FLAGS.n_epochs,
                                        n_features_nodes = sum(kept_indices_manual_selection), 
                                        n_features_global=config.NUM_GLOB_FEATURES, 
                                        layer_1=FLAGS.layer_1, 
                                        layer_2=FLAGS.layer_2,
                                        layer_linear = FLAGS.layer_linear,
                                        lstm_second=FLAGS.second_layer_memory) 
                                        #memory_network_block = nn.LSTM if FLAGS.memory_type=='LSTM' else nn.GRU,
                                        #lr=0.05,
                                        #teen_group=FLAGS.teen_group,
                                        #feat_sel=feature_sel_dict)
    #dict_double = utils.getDoubleDict(val_result)
    #dict_test = utils.getDoubleDict(test_result)
    #utils.writeToGoogleSheets(FLAGS.project_config_name, FLAGS.google_sheet_id-1, run_id, FLAGS.h_mess_arch_1, FLAGS.h_node_arch_1, FLAGS.h_mess_arch_2, FLAGS.h_node_arch_2, split_type, FLAGS.h_lookback, FLAGS.h_SEED, feature_sel_dict, dict_double, dict_test, dataset=training_dataset)                                      
    wandb.finish()
    return path, val_result, test_result, num_params, run_id


def main(argv):

    feature_sel_dict = {
        'speechAmount': False, 
        'isSpeaking': False, 
        'loudness':False, 
        'mfcc': False, 
        'energy': False, 
        'pitch': False, 
        '1stEnergy': False, 
        '1stPitch': False, 
        'mfcc_Std': False, 
        'energy_Std': False, 
        'pitch_Std': False, 
        '1stEnergy_Std': False, 
        '1stPitch_Std': False
    }

    FLAGS = absl.flags.FLAGS
    selected_indices = [int(x) for x in FLAGS.selected_indices.split(';')]
    FLAGS.selected_indices = selected_indices
    #FLAGS.logging.project=FLAGS.project_config_name

    variant = utils.get_user_flags(FLAGS, FLAGS_DEF)
    print(variant)
    # Hyperparameters


    # Prepare the dataset
    utils.set_random_seed(FLAGS.h_SEED)
    # mixing_episodes is the parameter that determines if we split train test valid data by episodes or in the whole dataset


    # In[164]:

    print("Starting training")
    list_features = list(config.FEATURE_DICT.keys())
    features_selected_temp = []
    for feature_id in selected_indices:
        features_selected_temp.append(list_features[feature_id])
    print("Running on", features_selected_temp)
    feature_sel_dict, kept_indices_manual_selection = utils.createFeatureBoolDict(features_selected_temp, feature_sel_dict, config.FEATURE_DICT)
    
    manual_name_filter = 'manual_'+'_'.join(features_selected_temp)
    #manual_name_filter = 'norm_manual_'+'_'.join(features_selected_temp)
    print(manual_name_filter)
    dataset_org=SummerschoolDataset("../training_data", "allepisodes_norm_no_permutations_norm.csv", 
    #dataset_org=SummerschoolDataset("../training_data", "allepisodes_t_t_no_permutations_norm.csv", 
                                    lookback=FLAGS.h_lookback, 
                                    kept_indices=kept_indices_manual_selection, 
                                    name_filter=manual_name_filter, 
                                    use_samples=FLAGS.use_samples,
                                    augment_mfcc=FLAGS.augment)
    utils.set_random_seed(FLAGS.h_SEED)
    path, val_result, test_result, num_params, run_id_full = get_data_and_train(dataset_org, kept_indices_manual_selection, feature_sel_dict, 'Full', FLAGS)
    dict_double_val = utils.getDoubleDict(val_result)
    dict_double_test = utils.getDoubleDict(test_result)
    utils.writeToGoogleSheetsSimple(FLAGS.project_config_name, FLAGS.google_sheet_id-1, run_id_full, 
                        FLAGS.layer_1, FLAGS.layer_2, FLAGS.second_layer_memory, FLAGS.layer_linear, 
                        'Mixed', FLAGS.h_SEED, dict_double_val, dict_double_test, 
                        val_result[0]['val_acc'], test_result[0]['test_acc'], num_params, 
                        FLAGS.h_BATCH_SIZE, FLAGS.loss_module,  ','.join([str(index) for index in FLAGS.selected_indices]),
                        'Full', 
                        FLAGS.group_embedding_dim, 'min', 'not applicable', 
                        path=path)  

    utils.set_random_seed(FLAGS.h_SEED)
    path_r2d2, val_result, test_result, num_params, run_id_r2d2 = get_data_and_train(dataset_org, kept_indices_manual_selection, feature_sel_dict, 'R2D2', FLAGS, finetune=True, path=path)
    dict_double_val = utils.getDoubleDict(val_result)
    dict_double_test = utils.getDoubleDict(test_result)
    utils.writeToGoogleSheetsSimpleCrossVal(FLAGS.project_config_name, FLAGS.google_sheet_id, run_id_r2d2, run_id_full,
                            FLAGS.layer_1, FLAGS.layer_2, FLAGS.second_layer_memory, FLAGS.layer_linear,
                            'Mixed', FLAGS.h_SEED, dict_double_val, dict_double_test, 
                            val_result[0]['val_acc'], test_result[0]['test_acc'], num_params, 
                            FLAGS.h_BATCH_SIZE, FLAGS.loss_module,  ','.join([str(index) for index in FLAGS.selected_indices]),
                            'R2D2', 
                            FLAGS.group_embedding_dim, 'min', path_r2d2)                          
    utils.set_random_seed(FLAGS.h_SEED)
    path_walle, val_result, test_result, num_params, run_id_walle = get_data_and_train(dataset_org, kept_indices_manual_selection, feature_sel_dict, 'WALLE', FLAGS, finetune=True, path=path)
    dict_double_val = utils.getDoubleDict(val_result)
    dict_double_test = utils.getDoubleDict(test_result)
    utils.writeToGoogleSheetsSimpleCrossVal(FLAGS.project_config_name, FLAGS.google_sheet_id, run_id_walle, run_id_full,
                            FLAGS.layer_1, FLAGS.layer_2, FLAGS.second_layer_memory, FLAGS.layer_linear,
                            'Mixed', FLAGS.h_SEED, dict_double_val, dict_double_test, 
                            val_result[0]['val_acc'], test_result[0]['test_acc'], num_params, 
                            FLAGS.h_BATCH_SIZE, FLAGS.loss_module,  ','.join([str(index) for index in FLAGS.selected_indices]),
                            'WALLE', 
                            FLAGS.group_embedding_dim, 'min', path_walle)   

    utils.set_random_seed(FLAGS.h_SEED)
    path_icub, val_result, test_result, num_params, run_id_icub = get_data_and_train(dataset_org, kept_indices_manual_selection, feature_sel_dict, 'ICUB', FLAGS, finetune=True, path=path)
    dict_double_val = utils.getDoubleDict(val_result)
    dict_double_test = utils.getDoubleDict(test_result)
    utils.writeToGoogleSheetsSimpleCrossVal(FLAGS.project_config_name, FLAGS.google_sheet_id, run_id_icub, run_id_full,
                            FLAGS.layer_1, FLAGS.layer_2, FLAGS.second_layer_memory, FLAGS.layer_linear,
                            'Mixed', FLAGS.h_SEED, dict_double_val, dict_double_test, 
                            val_result[0]['val_acc'], test_result[0]['test_acc'], num_params, 
                            FLAGS.h_BATCH_SIZE, 'MSELoss',  ','.join([str(index) for index in FLAGS.selected_indices]),
                            'ICUB', 
                            FLAGS.group_embedding_dim, 'min', path_icub)   

if __name__ == '__main__':
    absl.app.run(main)
