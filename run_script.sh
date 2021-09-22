#!/bin/bash

python main.py --gpu_id=0 \
--dataset_dir='/usr/sci/projs/DeepLearning/Cuong_Dataset/Nuclear_Forensics_Data/Synthetic_Routes_Magnifications/Magnification_v2' \
--datafile=all \
--num_epochs=45 --ep_init=15 \
--bz=16 \
--lr=2e-4 \
--momentum=0.9 \
--type_network=shared --resnet=resnet34 \
--fold_num=0 \
--saved_model_dir=saved_models \
--mags=10000x,25000x,50000x,100000x \
--isTesting

