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
    project_config_name='',
    google_sheet_id=7, # give ID of sheet starting with 1
    wandb_project_name="Brainstorming_Experimental_alt1_v0_2_3_linear",
    use_pretrained=False,
    pretrained_path="",
    loss_module='MSELoss',
    arch='RandomForest',
    do_test=True,
    group_size_training='2-3',
    limit_train=False,
    num_folds = 6,
    pooling_operation='min',
    n_estimators=100,
    max_depth=2,
    min_samples_split = 2,
    min_samples_leaf = 1,
    max_features = 'auto',
    bootstrap=True,
    criterion = 'gini',
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
    params = {  'n_estimators': FLAGS.n_estimators,
                'max_depth':FLAGS.max_depth,
                'min_samples_split': FLAGS.min_samples_split,
                'min_samples_leaf': FLAGS.min_samples_leaf,
                'max_features': FLAGS.max_features,
                'bootstrap': FLAGS.bootstrap,
                'criterion': FLAGS.criterion
            }
    print("Starting training")
    dataset_org=BrainstormingDatasetFixedSize("../training_data/", "annotated_data_brainstorming/", aggr_func=FLAGS.pooling_operation)
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

        
        
        
        
        # model, result, val_result, test_result, run_id, num_params = utils.train_graph_classifier(model_name="Brainstorming-RandomForest-Who", wandb_project_name=FLAGS.wandb_project_name, 
        #                                     h_SEED=FLAGS.h_SEED, device=FLAGS.device,
        #                                     graph_train_loader=graph_train_loader, graph_val_loader=graph_val_loader,
        #                                     graph_test_loader=graph_test_loader,
        #                                     early_stopping=FLAGS.early_stopping,
        #                                     tune_lr=FLAGS.tune_learning_rate,
        #                                     max_epochs=FLAGS.n_epochs,
        #                                     loss_module=loss_module,
        #                                     limit_train = FLAGS.limit_train)
    dict_double_val = utils.getDoubleDict(val_result)
    dict_double_test = utils.getDoubleDict(test_result)
    run_id = get_run_id(10)
    NAME_SCALER_FILE = "../scalers/"+FLAGS.project_config_name+"-"+split_type+"-"+run_id+".pkl"
    #pickle.dump(scaler_n, open(NAME_SCALER_FILE, 'wb'))
    

    utils.writeToGoogleSheetsTradLinear(FLAGS.project_config_name, FLAGS.google_sheet_id-1, run_id, dict_double_val,dict_double_test,val_result[0]['val_acc'], test_result[0]['test_acc'],
                            FLAGS.arch, params,
                            split_type, FLAGS.h_SEED,
                            FLAGS.limit_train if not FLAGS.limit_train else 'Groups of '+str(FLAGS.group_size_training)+' , #Ep: '+str(FLAGS.limit_amount_of_train_val),
                            FLAGS.pooling_operation, use_drive=FLAGS.use_drive)
    if FLAGS.do_test:
        dataset_org=BrainstormingDatasetFixedSize("../training_data/raw", "data_aggregated_four/", test_data=True, aggr_func='min', group_sizes_train="4")
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
        utils.writeToGoogleSheetsTradLinear(FLAGS.project_config_name, FLAGS.google_sheet_id, run_id, dict_double_val,dict_double_test,val_result[0]['val_acc'], test_result[0]['test_acc'],
                            FLAGS.arch, params,
                            split_type, FLAGS.h_SEED,
                            FLAGS.limit_train if not FLAGS.limit_train else 'Groups of '+str(FLAGS.group_size_training)+' , #Ep: '+str(FLAGS.limit_amount_of_train_val),
                            FLAGS.pooling_operation, use_drive=FLAGS.use_drive)

if __name__ == '__main__':
    absl.app.run(main)
