import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import os
import math
import random
from glob import glob
import os.path as osp
import pickle


class SPNDataset(data.Dataset):
    def __init__(self, is_test=False, cfg=None):

        self.is_test= is_test
        self.cfg = cfg
        self.init_seed = False
        self.training_data_dir =  "/home/qym/datasets/training_data_count_3model.pkl"
        # self.test_data_dir = self.cfg.dataset_batch
        self.test_data_dir = "/home/qym/datasets/training_data_count_3model.pkl"
        # count: testSetCount_norm_1000-mean_crop3.pkl
        self.data_list = []
        
        # load all data into memory
        self.load_data()

    def load_data(self):
        if self.is_test:
            data_file = self.test_data_dir
            print("[Loading test data from {}]".format(data_file))
        else:
            data_file = self.training_data_dir
            print("[Loading training data from {}]".format(data_file))
        
        with open(data_file, 'rb') as f:
            self.data_list = pickle.load(f)[210000:220000]
        #print("len(data_list):", len(self.data_list))

    def __getitem__(self, index):
        if not self.init_seed:
            # workers get different random seed
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                #print(worker_info.id)
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True
        
        index = index % len(self.data_list)     
        spn_num = len(self.cfg.spn_input_dims)
        if self.cfg.base.useTree:
            spn_num = int(spn_num / 2)

        spn_values = None
        if self.cfg.base.useTree:
            tmp = []
            for i in range(spn_num):
                flatTree = self.data_list[index][i][0]
                #print(flatTree)
                flatTree = flatTree.reshape(-1)
                #print("flatTree.shape:", flatTree.shape)
                if isinstance(flatTree, np.ndarray):
                    flatTree = torch.zeros(self.cfg.base.spn_input_dims[i*2])
                    #print(flatTree.shape)
                #print(type(flatTree), i)
                tmp.append(flatTree)

                indexes = self.data_list[index][i][1]
                indexes = indexes.reshape(-1)
                #print(type(indexes), i)
                #if isinstance(indexes, np.ndarray):
                #    print("indexes:",indexes)
                
                tmp.append(indexes)
            
            spn_values = torch.cat(tmp)
            #print(spn_values.shape)
        else:
            spn_values = np.array(sum(self.data_list[index][:spn_num], []))
            #print("spn_values:", spn_values)
            spn_values = torch.from_numpy(spn_values).float()
        
        spn_preds = np.array(self.data_list[index][-2])
        gt = np.array([self.data_list[index][-1]])

        
        spn_preds = torch.from_numpy(spn_preds).float()
        gt = torch.from_numpy(gt).float()

        return spn_values, spn_preds, gt

    def __len__(self):
        return len(self.data_list)

class SPNNormDataset(data.Dataset):
    def __init__(self, is_test=False, cfg=None):

        self.is_test= is_test
        self.cfg = cfg
        self.init_seed = False
        self.training_data_dir = "/home/qym/datasets/training_data_count_4model.pkl"
        self.test_data_dir = "/home/qym/datasets/testSetAvg_new.pkl"
        self.data_list = []
        
        # load all data into memory
        self.load_data()

    def load_data(self):
        if self.is_test:
            data_file = self.test_data_dir
            print("[Loading test data from {}]".format(data_file))
        else:
            data_file = self.training_data_dir
            print("[Loading training data from {}]".format(data_file))
        
        with open(data_file, 'rb') as f:
            self.data_list = pickle.load(f)

    def __getitem__(self, index):
        if not self.init_seed:
            # workers get different random seed
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                #print(worker_info.id)
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True
        
        index = index % len(self.data_list)     
        spn_num = len(self.cfg.spn_input_dims)
        if self.cfg.base.useTree:
            spn_num = int(spn_num / 2)

        spn_values = None
        if self.cfg.base.useTree:
            tmp = []
            for i in range(spn_num):
                flatTree = self.data_list[index][i][0]
                #print(flatTree)
                flatTree = flatTree.reshape(-1)
                #print("flatTree.shape:", flatTree.shape)
                if isinstance(flatTree, np.ndarray):
                    flatTree = torch.zeros(self.cfg.base.spn_input_dims[i*2])
                    #print(flatTree.shape)
                #print(type(flatTree), i)
                tmp.append(flatTree)

                indexes = self.data_list[index][i][1]
                indexes = indexes.reshape(-1)
                #print(type(indexes), i)
                #if isinstance(indexes, np.ndarray):
                #    print("indexes:",indexes)
                
                tmp.append(indexes)
            
            spn_values = torch.cat(tmp)
            #print(spn_values.shape)
        else:
            spn_values = np.array(sum(self.data_list[index][:spn_num], []))
            spn_values = torch.from_numpy(spn_values).float()
        
        spn_preds = np.array(self.data_list[index][-2])
        gt = np.array([self.data_list[index][-1]])

        
        spn_preds = torch.from_numpy(spn_preds).float()
        gt = torch.from_numpy(gt).float()

        return spn_values, spn_preds, gt

    def __len__(self):
        return len(self.data_list)

def fetch_dataloader(args):
    """ Create the data loader for the corresponding trainign set """

    train_dataset = SPNDataset(is_test=False, cfg=args)

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
        pin_memory=False, shuffle=True, num_workers=1, drop_last=True)

    print('Training with %d data' % len(train_dataset))
    return train_loader
        
def fetch_test_dataloader(args):
    """ Create the data loader for the corresponding trainign set """

    test_dataset = SPNDataset(is_test=True, cfg=args)

    test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, 
        pin_memory=False, shuffle=True, num_workers=1, drop_last=True)

    print('Testing with %d data' % len(test_dataset))
    return test_loader