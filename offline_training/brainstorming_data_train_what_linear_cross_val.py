#!/usr/bin/env python
# coding: utf-8

# In[137]:


import utils
import config
import wandb
import torch
from dataset import SummerschoolDataset, BrainstormingDatasetFixedSize
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
    h_BATCH_SIZE = 64,
    h_SEED = 0,
    memory_type='LSTM',
    early_stopping=False,
    tune_learning_rate=False,
    n_epochs=100,
    group_embedding_dim=6, # was 12 for all other runs
    project_config_name='BrainstormingExplore_Limited_Training_Data_experiment_alt1_v0_linear',
    google_sheet_id=1, # give ID of sheet starting with 1
    wandb_project_name="Brainstorming_Experimental_alt1_v0_2_3_linear",
    use_pretrained=False,
    pretrained_path="",
    loss_module='MSELoss',
    arch='16-32-8',
    group_size_training='3',
    limit_train=False,
    limit_amount_of_train_val=12,
    limit_amount_test = 6,
    num_folds = 6,
    pooling_operation='min',
    dropout=0.0,
    do_test=True,
    use_drive=False,
)







def main(argv):


    FLAGS = absl.flags.FLAGS

    variant = utils.get_user_flags(FLAGS, FLAGS_DEF)
    print(variant)
 
    utils.set_random_seed(FLAGS.h_SEED)
  
    loss_module = nn.MSELoss()
    if FLAGS.loss_module == "CrossEntropyLoss":
        loss_module = nn.CrossEntropyLoss()
    # In[164]:

    print("Starting training")
    dataset_org=BrainstormingDatasetFixedSize("../training_data/", "annotated_data_what_all/", aggr_func = 'min', type='what')
    utils.set_random_seed(FLAGS.h_SEED)
    data_over_folds = {'ident':'', 'num_decisions':[],'num_params': 0,'val': {'acc':[], 'macro avg: f1':[]},'test': {'acc':[], 'macro avg: f1':[]}}
    common_ident = ''
    data_over_folds_test = {'ident':'', 'num_decisions':[],'num_params': 0,'val': {'acc':[], 'macro avg: f1':[]},'test': {'acc':[], 'macro avg: f1':[]}}
    data_over_folds_test['path'] = []
    data_over_folds_test['scaler'] = []
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
        scaler_n = preprocessing.StandardScaler().fit(train_dataset.data.x)
        train_dataset.transform_scaler = scaler_n
        train_dataset.activate_transform()
        valid_dataset.transform_scaler = scaler_n
        valid_dataset.activate_transform()
        test_dataset.transform_scaler = scaler_n
       
        graph_train_loader, graph_val_loader, graph_test_loader = utils.loadData(train_dataset, valid_dataset, test_dataset, FLAGS.h_BATCH_SIZE)
        model, result, val_result, test_result, run_id, num_params, path = utils.train_graph_classifier(model_name="Brainstorming-Linear-What", wandb_project_name=FLAGS.wandb_project_name, 
                                            h_SEED=FLAGS.h_SEED, device=FLAGS.device,
                                            graph_train_loader=graph_train_loader, graph_val_loader=graph_val_loader,
                                            graph_test_loader=graph_test_loader,
                                            return_path=True,
                                            early_stopping=FLAGS.early_stopping,
                                            tune_lr=FLAGS.tune_learning_rate,
                                            max_epochs=FLAGS.n_epochs,
                                            loss_module=loss_module,
                                            architecture=FLAGS.arch,
                                            n_output_dim_node = 2, 
                                            n_output_dim_action = 4, 
                                            n_features = int(config.NUM_NODE_FEATURES)*2, 
                                            split_mode = split_type, 
                                            limit_train = FLAGS.limit_train,
                                            dropout_p = FLAGS.dropout,
                                            lr=0.1)
        dict_double_val = utils.getDoubleDict(val_result)
        dict_double_test = utils.getDoubleDict(test_result)
        NAME_SCALER_FILE = "../scalers/"+FLAGS.project_config_name+"-"+split_type+"-"+run_id+".pkl"
        #pickle.dump(scaler_n, open(NAME_SCALER_FILE, 'wb'))
        if common_ident == '':
            common_ident = run_id
            data_over_folds['ident'] = run_id
            data_over_folds['num_params'] = num_params
            data_over_folds_test['ident'] = run_id
            data_over_folds_test['num_params'] = num_params
        # utils.writeToGoogleSheetsLinearCrossVal(FLAGS.project_config_name, FLAGS.google_sheet_id, run_id, common_ident,
        #                     FLAGS.arch,
        #                     split_type, FLAGS.h_SEED, dict_double_val, dict_double_test, 
        #                     val_result[0]['val_acc'], test_result[0]['test_acc'], num_params, 
        #                     FLAGS.h_BATCH_SIZE, FLAGS.loss_module, len(train_dataset.data.y_who),
        #                     FLAGS.limit_train if not FLAGS.limit_train else 'Groups of '+str(FLAGS.group_size_training)+' , #Ep: '+str(FLAGS.limit_amount_of_train_val),
        #                     FLAGS.pooling_operation, FLAGS.dropout, use_drive=FLAGS.use_drive)                                      
        data_over_folds['num_decisions'].append(len(train_dataset.data.y_who))
        data_over_folds['val']['acc'].append(val_result[0]['val_acc'])
        data_over_folds['test']['acc'].append(test_result[0]['test_acc'])
        data_over_folds['test']['macro avg: f1'].append(dict_double_test['macro avg']['f1'])
        data_over_folds['val']['macro avg: f1'].append(dict_double_val['macro avg']['f1'])
        wandb.finish()
        if FLAGS.do_test:
            data_over_folds_test['path'].append(path)
            data_over_folds_test['scaler'].append(scaler_n)  
    utils.writeToGoogleSheetsCrossValSummaryLinear(FLAGS.project_config_name, FLAGS.google_sheet_id-1, data_over_folds,
                            FLAGS.arch,
                            split_type, FLAGS.h_SEED, FLAGS.h_BATCH_SIZE, FLAGS.loss_module,
                            FLAGS.limit_train if not FLAGS.limit_train else 'Groups of '+str(FLAGS.group_size_training)+' , #Ep: '+str(FLAGS.limit_amount_of_train_val),
                            FLAGS.pooling_operation, FLAGS.dropout,use_drive=FLAGS.use_drive)
    if not FLAGS.do_test:
        return
    for i in range(len(data_over_folds_test['path'])):
        path = data_over_folds_test['path'][i]
        dataset_org=BrainstormingDatasetFixedSize("../training_data/raw", "data_aggregated_four_what/", test_data=True, group_sizes_train="4", aggr_func = 'min', type='what')
        utils.set_random_seed(FLAGS.h_SEED)
        dataset_org.transform_scaler = data_over_folds_test['scaler'][i]
        dataset_org.activate_transform()
        split_type="None"

        graph_train_loader, graph_val_loader, graph_test_loader = utils.loadData(dataset_org, dataset_org, dataset_org, FLAGS.h_BATCH_SIZE)
        print("Loading from", path)
        model, result, val_result, test_result, run_id, num_params = utils.train_graph_classifier(model_name="Brainstorming-Linear-What", wandb_project_name=FLAGS.wandb_project_name, 
                                            h_SEED=FLAGS.h_SEED, device=FLAGS.device,
                                            graph_train_loader=graph_train_loader, graph_val_loader=graph_val_loader,
                                            graph_test_loader=graph_test_loader,
                                            load_pretrained=True,
                                            pretrained_filename=path,
                                            early_stopping=FLAGS.early_stopping,
                                            tune_lr=FLAGS.tune_learning_rate,
                                            max_epochs=FLAGS.n_epochs,
                                            loss_module=loss_module,
                                            architecture=FLAGS.arch,
                                            n_output_dim_node = 2, 
                                            n_output_dim_action = 4, 
                                            n_features = int(config.NUM_NODE_FEATURES)*2, 
                                            split_mode = split_type, 
                                            limit_train = FLAGS.limit_train,
                                            dropout_p = FLAGS.dropout,
                                            lr=0.1)
        dict_double_val = utils.getDoubleDict(val_result)
        dict_double_test = utils.getDoubleDict(test_result)
        data_over_folds_test['val']['acc'].append(val_result[0]['val_acc'])
        data_over_folds_test['test']['acc'].append(test_result[0]['test_acc'])
        data_over_folds_test['test']['macro avg: f1'].append(dict_double_test['macro avg']['f1'])
        data_over_folds_test['val']['macro avg: f1'].append(dict_double_val['macro avg']['f1'])
    utils.writeToGoogleSheetsCrossValSummaryLinear(FLAGS.project_config_name, FLAGS.google_sheet_id, data_over_folds_test,
                        FLAGS.arch,
                        split_type, FLAGS.h_SEED, FLAGS.h_BATCH_SIZE, FLAGS.loss_module,
                        FLAGS.limit_train if not FLAGS.limit_train else 'Groups of '+str(FLAGS.group_size_training)+' , #Ep: '+str(FLAGS.limit_amount_of_train_val),
                        FLAGS.pooling_operation, FLAGS.dropout, use_drive=FLAGS.use_drive)


if __name__ == '__main__':
    absl.app.run(main)
