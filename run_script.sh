#!/bin/bash

python main.py --gpu_id=0 \
--datafile=50k100k \
--num_epochs=45 --ep_init=15 \
--bz=16 \
--lr=2e-4 \
--momentum=0.9 \
--type_network=shared --resnet=resnet34 \
--fold_num=0 \
--saved_model_dir=saved_models \
--isTesting
