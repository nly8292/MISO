#!/home/sci/nly8292/document/pytorch_env_dgx/bin/python

#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH -o slurm-%j.out-%N
#SBATCH -e slurm-%j.err-%N
#SBATCH  --gres=gpu:1

from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np
from skimage import io
from PIL import Image as PILimage
import time
import os
import scipy
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import copy
import argparse
import sys
import time

from torch.utils.data import DataLoader

from model import *
from dataset import *
from utils import *

def Main():
    
    parser = argparse.ArgumentParser()  
    parser.add_argument('--gpu_id', type=int, default = 0, help = 'GPU#')
    parser.add_argument('--isTesting', action="store_true", help='Specify the process')      
    parser.add_argument('--datafile_dir', type=str, default = '.', help = 'Directory of datafile')      
    parser.add_argument('--datafile', type=str, default = '', help = 'Core name of data file to be created/loaded')
    parser.add_argument('--num_epochs', type=int, default = 10, help = 'Number of epochs')
    parser.add_argument('--ep_init', type=int, default = 15, help = 'Number of epochs for initial training')
    parser.add_argument('--bz', type=int, default = 16, help = 'Batch size')    
    parser.add_argument('--lr', type=float, default = 2e-4, help = 'Learning rate')
    parser.add_argument('--momentum', type=float, default = 0.9, help = 'Momentum')
    parser.add_argument('--type_network', type=str, default = 'shared', help = 'Type of network')
    parser.add_argument('--resnet', type=str, default = 'resnet34', help = 'Type of resnet')
    parser.add_argument('--fold_num', type=int, default = 0, help = 'Fold number')
    parser.add_argument('--saved_model_dir', type=str, default = './saved_models', help = 'Directory of saved models')
    parser.add_argument('--create_trainval_file', action="store_true", help='Create train/val text files') 
    parser.add_argument('--mags', type=str, default = '10000x,25000x,50000x,100000x', help='Magnifications used for classification')
    parser.add_argument('--trainval_file_corename', type=str, default = '', help='Core name of train/val text file to be created')

    params, unparsed = parser.parse_known_args()

    os.environ["CUDA_VISIBLE_DEVICES"]= str(params.gpu_id)

    params.mags = [x for x in params.mags.split(',')]
    if params.create_trainval_file:
        create_trainval_file('/usr/sci/projs/DeepLearning/Cuong_Dataset/Nuclear_Forensics_Data/Synthetic_Routes_Magnifications/Magnification_v2',\
         params.fold_num, params.mags, params.datafile_dir, params.datafile) 

    core_datafile_name = '%s_fold%d' %(params.datafile, params.fold_num)    

    if not params.isTesting:
        dataloader_list = ['train', 'val']
    else:
        dataloader_list = ['val']    
        
    dataloaders_dict = {}

    for x in dataloader_list:
        since = time.time()             
        
        datafile = '%s/%s_%s.txt' %(params.datafile_dir,x,core_datafile_name)
        
        curr_dataset = SynRoutes(datafile, x)
        dataloaders_dict[x] = torch.utils.data.DataLoader(curr_dataset, batch_size=params.bz, shuffle=x=='train', num_workers=4)

        time_elapsed = time.time() - since
        print('--- Finish loading ' + x + ' data in {:.0f}m {:.0f}s---'.format(time_elapsed // 60, time_elapsed % 60))
        
    saved_file_name = '%s_%s_%s' %(core_datafile_name,params.type_network,params.resnet)
    
    num_classes = dataloaders_dict[dataloader_list[0]].dataset.get_num_classes()
    classes = dataloaders_dict[dataloader_list[0]].dataset.get_classes()
    mags = dataloaders_dict[dataloader_list[0]].dataset.get_mags()

    miso = Model(params, num_classes, dataloaders_dict[dataloader_list[0]].dataset.num_mags)     
    if not params.isTesting:                        
        logger = create_logger()
        ## Logging params ##
        log_params(logger,params,core_datafile_name, mags, classes)
        miso.train_model(dataloaders_dict,logger)
    else:
        miso.load_trained_model(params.saved_model_dir)

    miso.predict(dataloaders_dict['val'], classes)

Main()
