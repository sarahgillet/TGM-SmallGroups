#!/usr/bin/env python
# coding: utf-8

# In[137]:


import utils
import config
import wandb
import torch
from dataset import SummerschoolDataset, BrainstormingDatasetFixedSize, BrainstormingDatasetMinimalContext
import MMPN
import torch.nn as nn

import absl.app
import absl.flags
from sklearn import preprocessing
import pickle
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import torch_scatter
import numpy as np
import string
import random
import pygsheets
import pandas as pd

utils.set_random_seed(42)

FLAGS_DEF = utils.define_flags_with_default(
    device= 'cpu',
    h_lookback= 10,
    use_samples=1,
    h_TEST_SIZE= 0.1,
    h_VALID_SIZE= 1.0/6.0,
    h_BATCH_SIZE = 1500,
    h_SEED = 0,
    memory_type='LSTM',
    early_stopping=False,
    tune_learning_rate=False,
    n_epochs=1,
    group_embedding_dim=6, # was 12 for all other runs
    project_config_name='BrainstormingExplore_Limited_Training_Data_experiment_linear_trad_alt1_v1_test',
    google_sheet_id=1, # give ID of sheet starting with 1
    wandb_project_name="Brainstorming_Experimental_alt1_v0_2_3_linear",
    use_pretrained=False,
    pretrained_path="",
    loss_module='MSELoss',
    arch='RandomForest',
    group_size_training='2-3',
    limit_train=False,
    limit_amount_of_train_val=12,
    limit_amount_test = 6,
    num_folds = 6,
    pooling_operation='min',
    n_estimators=100,
    max_depth=2,
    min_samples_split = 2,
    min_samples_leaf = 1,
    max_features = 'auto',
    bootstrap=True,
    criterion = 'gini',
    do_test=False,
    use_drive=False


)

def forward(model, data):
    x, y = data.x, data.y_who_local
    x_np = x.numpy()
    y_np = y.numpy()
    y_pred = model.predict_proba(x_np)
    preds_val, preds_batch_id = torch_scatter.scatter_max(torch.Tensor(y_pred[:,0]), data.batch, dim=0)
    #     #print(preds_val.shape)
    target_ids_batches = []
    num_nodes = torch.Tensor(np.array(data.num_nodes_loc)).to(torch.int64).to(device=x.device).squeeze(1)
    for n in num_nodes:
        target_ids_batches.extend(np.arange(n))
    
    index_preds_batch = np.array(preds_batch_id.squeeze()).astype(int)
    preds = torch.tensor(np.array(target_ids_batches)[index_preds_batch])
    #loss = self.loss_module(torch.Tensor([1]),torch.Tensor([0]))
    return preds
    

def build_output(dict_report, acc, mode):
    report = {}
    for key in dict_report:
        if type(dict_report[key]) == dict:
            for key_small in dict_report[key]:
                report[mode+'-'+key+'-'+key_small] = dict_report[key][key_small]
        else:
            report[mode+'-'+key] = dict_report[key]
    report[mode+'_acc'] = acc
    return report

def get_run_id(length):
    characters = string.ascii_letters + string.digits
    
    # Generate the random string
    random_string = ''.join(random.choice(characters) for _ in range(length))
    return random_string

def didParamsRunToDf(params, df):
    df_worksheet_sel_in_config = df
    for key in params:
        # find the row in df_worksheet_sel that is not in df_config_sel
        if key=='bootstrap':
            #print(df_worksheet_sel_in_config[key].astype(str), str(params[key]).upper())
            df_worksheet_sel_in_config = df_worksheet_sel_in_config[df_worksheet_sel_in_config[key].astype(str)==str(params[key]).upper()]
        elif key=='Training Set':
            print(df_worksheet_sel_in_config[key].astype(str))
            print(str(params[key]))
        else:
            df_worksheet_sel_in_config = df_worksheet_sel_in_config[df_worksheet_sel_in_config[key].astype(str)==str(params[key])]
        print(len(df_worksheet_sel_in_config), key)
        #if len(df_worksheet_sel_in_config) < 30:
        #    print(df_worksheet_sel_in_config[elemnts_of_interest])   
        # compare the two dataframes
    if len(df_worksheet_sel_in_config) == 0:
        #print('Row not found in df_worksheet_sel_in_config')
        return False
    return True


def main(argv):
    FLAGS = absl.flags.FLAGS
    FLAGS.group_size_training = FLAGS.group_size_training.replace("\r", "")

    variant = utils.get_user_flags(FLAGS, FLAGS_DEF)
    print(variant)

    loss_module = nn.MSELoss()
    if FLAGS.loss_module == "CrossEntropyLoss":
        loss_module = nn.CrossEntropyLoss()

    params = {  'n_estimators': FLAGS.n_estimators,
                'max_depth':FLAGS.max_depth,
                'min_samples_split': FLAGS.min_samples_split,
                'min_samples_leaf': FLAGS.min_samples_leaf,
                'max_features': FLAGS.max_features,
                'bootstrap': FLAGS.bootstrap,
                'criterion': FLAGS.criterion
            }

    utils.set_random_seed(FLAGS.h_SEED)
    print("Starting training")
    dataset_org=BrainstormingDatasetFixedSize("../training_data/", "annotated_data_brainstorming/", aggr_func=FLAGS.pooling_operation)
    
    utils.set_random_seed(FLAGS.h_SEED)
    data_over_folds = {'ident':'', 'num_decisions':[],'num_params': 0,'val': {'acc':[], 'macro avg: f1':[]},'test': {'acc':[], 'macro avg: f1':[]}}
    common_ident = ''
    data_over_folds_test = {'ident':'', 'num_decisions':[],'num_params': 0,'val': {'acc':[], 'macro avg: f1':[]},'test': {'acc':[], 'macro avg: f1':[]}}
    data_over_folds_test['model'] = []
    data_over_folds_test['scaler'] = []
    print(dataset_org.data.x.shape)
    if FLAGS.limit_train:
            train_valid_datasets, test_dataset, split_type = utils.generateTrainTestValidDataset(
                dataset_org, FLAGS.h_SEED, FLAGS.h_TEST_SIZE, FLAGS.h_VALID_SIZE, mixing_episodes=False, 
                fixed_group_size_training=FLAGS.group_size_training, 
                dataset_size_train_val=FLAGS.limit_amount_of_train_val, dataset_size_test=FLAGS.limit_amount_test, folds=True, num_folds=FLAGS.num_folds)
    else:
        raise NotImplementedError

    for fold in range(FLAGS.num_folds):
        run_id = get_run_id(10)
        train_dataset = train_valid_datasets[fold][0]
        valid_dataset = train_valid_datasets[fold][1]
        scaler_n = preprocessing.StandardScaler().fit(train_dataset.data.x)
        train_dataset.transform_scaler = scaler_n
        train_dataset.activate_transform()
        valid_dataset.transform_scaler = scaler_n
        valid_dataset.activate_transform()
        test_dataset.transform_scaler = scaler_n
 
        graph_train_loader, graph_val_loader, graph_test_loader = utils.loadData(train_dataset, valid_dataset, test_dataset, FLAGS.h_BATCH_SIZE)
        print(graph_train_loader)
        data = next(iter(graph_train_loader))
        x, y = data.x, data.y_who_local
        print(data.batch)
        # Convert inputs to numpy arrays
        x_np = x.numpy()
        y_np = y.numpy()
        # Train the model
        model = RandomForestClassifier(n_estimators=FLAGS.n_estimators, max_depth=FLAGS.max_depth if FLAGS.max_depth>=0 else None, 
                                    criterion=FLAGS.criterion, min_samples_split=FLAGS.min_samples_split, 
                                    min_samples_leaf=FLAGS.min_samples_leaf, max_features=FLAGS.max_features,
                                    bootstrap=FLAGS.bootstrap,random_state=FLAGS.h_SEED)
        model.fit(x_np, y_np)
        # Calculate loss
        y_pred = forward(model, data)
        accuracy = accuracy_score(data.y_who, y_pred)
        print("Training accuracy", accuracy)
        #self.log('train_loss', loss)
        #self.log('train_acc', accuracy)
        data = next(iter(graph_val_loader))
        y_pred = forward(model, data)
        accuracy = accuracy_score(data.y_who, y_pred)
        
        print("Validation accuracy", accuracy)
        dict_report = classification_report(data.y_who, y_pred, output_dict=True, zero_division=0)
        print(dict_report)
        val_result = [build_output(dict_report, accuracy, 'val')]
        data = next(iter(graph_test_loader))
        y_pred = forward(model, data)
        accuracy = accuracy_score(data.y_who, y_pred)
        
        print("Test accuracy", accuracy)
        dict_report = classification_report(data.y_who, y_pred, output_dict=True, zero_division=0)
        print(dict_report)
        test_result = [build_output(dict_report, accuracy, 'test')]


        dict_double_val = utils.getDoubleDict(val_result)
        dict_double_test = utils.getDoubleDict(test_result)
        NAME_SCALER_FILE = "../scalers/"+FLAGS.project_config_name+"-"+split_type+"-"+run_id+".pkl"
        

        if common_ident == '':
            common_ident = run_id
            data_over_folds['ident'] = run_id
            data_over_folds_test['ident'] = run_id
                                    
        data_over_folds['num_decisions'].append(len(train_dataset.data.y_who))
        data_over_folds['val']['acc'].append(val_result[0]['val_acc'])
        data_over_folds['test']['acc'].append(test_result[0]['test_acc'])
        data_over_folds['test']['macro avg: f1'].append(dict_double_test['macro avg']['f1'])
        data_over_folds['val']['macro avg: f1'].append(dict_double_val['macro avg']['f1'])
        wandb.finish()
        if FLAGS.do_test:
            data_over_folds_test['model'].append(model)
            data_over_folds_test['scaler'].append(scaler_n)
    utils.writeToGoogleSheetsCrossValSummaryTradLinear(FLAGS.project_config_name, FLAGS.google_sheet_id-1, data_over_folds,
                            'RandomForest', params,
                            split_type, FLAGS.h_SEED,
                            FLAGS.limit_train if not FLAGS.limit_train else 'Groups of '+str(FLAGS.group_size_training)+' , #Ep: '+str(FLAGS.limit_amount_of_train_val),
                            FLAGS.pooling_operation, use_drive=FLAGS.use_drive)
    if FLAGS.do_test:
        for model in data_over_folds_test['model']:
            dataset_org=BrainstormingDatasetFixedSize("../training_data/raw", "data_aggregated_four/", test_data=True, group_sizes_train="4", aggr_func = 'min')
            utils.set_random_seed(FLAGS.h_SEED)
            dataset_org.transform_scaler = scaler_n
            dataset_org.activate_transform()
            split_type="None"

            graph_train_loader, graph_val_loader, graph_test_loader = utils.loadData(dataset_org, dataset_org, dataset_org, FLAGS.h_BATCH_SIZE)
            #print("Loading from", path)
            data = next(iter(graph_train_loader))
            y_pred = forward(model, data)
            accuracy = accuracy_score(data.y_who, y_pred)
            dict_report = classification_report(data.y_who, y_pred, output_dict=True, zero_division=0)
            val_result = [build_output(dict_report, accuracy, 'val')]
            test_result = [build_output(dict_report, accuracy, 'test')]
            dict_double_val = utils.getDoubleDict(val_result)
            dict_double_test = utils.getDoubleDict(test_result)
            data_over_folds_test['num_decisions'].append(len(train_dataset.data.y_who))
            data_over_folds_test['val']['acc'].append(val_result[0]['val_acc'])
            data_over_folds_test['test']['acc'].append(test_result[0]['test_acc'])
            data_over_folds_test['test']['macro avg: f1'].append(dict_double_test['macro avg']['f1'])
            data_over_folds_test['val']['macro avg: f1'].append(dict_double_val['macro avg']['f1'])
        utils.writeToGoogleSheetsCrossValSummaryTradLinear(FLAGS.project_config_name, FLAGS.google_sheet_id, data_over_folds_test,
                            FLAGS.arch, params,
                            split_type, FLAGS.h_SEED,
                            FLAGS.limit_train if not FLAGS.limit_train else 'Groups of '+str(FLAGS.group_size_training)+' , #Ep: '+str(FLAGS.limit_amount_of_train_val),
                            FLAGS.pooling_operation, use_drive=FLAGS.use_drive)

if __name__ == '__main__':
    absl.app.run(main)
