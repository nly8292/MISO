
# Mixed Sample Synthesis
This is the official implementation of the multi-input single-output models proposed in "<a href="https://doi.org/10.1016/j.jnucmat.2020.152082">Determining uranium ore concentrates and their calcination products via image classification of multiple magnifications</a>"

## Setup

### Prerequisites
- Linux
- NVIDIA GPU + CUDA CuDNN (CPU mode may work without any modification, but untested)
- Python 3.6
- Install necessary libraries with pip
```
pip3 install -r requirements.txt
```


## Traing/Testing
Execute the script `run_script.sh` to train or test pretrained MISO models. Furthermore, the following params inside `run_script.sh` can be modified to obtain desired behavior.
- `gpu_id` - GPU ID
- `isTesting` - Switch to testing mode
- `create_trainval_file` - Initiate data file creation process
- `trainval_file_corename` - Core name of train/val text file to be created
- `datafile_dir` - Directory of datafile to be loaded
- `datafile` - Core name of data file to be loaded
- `num_epochs` - Number of training epochs
- `ep_init` - Number of epochs for the transfer learning stage
- `bz` - Batch size
- `lr` - Learning rate
- `momentum` - Momentum for SGD optimizer    
- `type_network` - Type of MISO model [`shared (Default) | unshared` ]
- `resnet` - Type of Resnet model [`resnet18 | resnet34 (Default) | resnet50`]
- `mags` - Magnifications used for classification
- `fold_num` - Fold-th to be used for training in the 5-fold cross validation    
- `saved_model_dir` - Directory for saving the model's weights
    

## Citation
If you find this code useful for your research, please cite our paper:

```
@article{MISO, 
     author={Ly, Cuong and Vachet, Clement and Schwerdt, Ian and Abbott, Erik and Brenkmann, Alexandria and Mcdonald, Luther W. and Tasdizen, Tolga},
     title={Determining uranium ore concentrates and their calcination products via image classification of multiple magnifications}, 
     volume={533}, 
     DOI={10.1016/j.jnucmat.2020.152082}, 
     journal={Journal of Nuclear Materials}, 
     year={2020}, 
     pages={152082}} 
```

