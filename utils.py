import scipy
import numpy as np
import matplotlib.pyplot as plt
import itertools
import glob
import random
import datetime
import os
import logging
                        
ROUTES = ['U3O8ADU', 'U3O8AUC', 'U3O8MDU', 'U3O8SDU', 'UO2ADU', 'UO2AUCd', 'UO2AUCi', 'UO2SDU',\
            'UO3ADU', 'UO3AUC', 'UO3MDU', 'UO3SDU']

def majority_vote(preds, labels):
    ''' 
    Compute the final prediction based on majority voting 
    '''

    ## 4-crop per image ##
    num_per_img = 4

    maj_vec = np.zeros((labels.shape[0]//num_per_img,))
    maj_labels = np.copy(maj_vec)

    for i in range(0,labels.shape[0],num_per_img):
        curr_mode,_ = scipy.stats.mode(preds[i:i+num_per_img])
        
        maj_vec[i//num_per_img] = curr_mode[0]
        maj_labels[i//num_per_img] = labels[i]

    acc = float(len(np.where(maj_vec == maj_labels)[0])) / len(maj_vec)
    print('Majority vote Acc = {:.6f}'.format(acc))

    return maj_vec, maj_labels

def plot_confusion_matrix(conf_matrix, classes, saved_exp_name,
                      normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    '''
    Display confusion matrix
    '''

    plt.figure()
    if normalize:
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(conf_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.xlim(-0.5,len(classes)-0.5)
    plt.ylim(len(classes)-0.5,-0.5)

    fmt = '.2f' if normalize else 'd'
    thresh = conf_matrix.max() / 2.
    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        plt.text(j, i, format(conf_matrix[i, j], fmt),
                 horizontalalignment="center",
                 fontsize=8,
                 color="white" if conf_matrix[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    if normalize:
        plt.savefig('./norm_confusion_matrix_%s.png' %saved_exp_name)
    else:
        plt.savefig('./confusion_matrix_%s.png' %saved_exp_name)


def create_trainval_file(rdir, fold_num, mags, trainval_file_dir, trainval_file_corename):
    '''
    Generate train/val text file of a dataset with a specified kth fold 
    '''

    total_fold = 5
    
    train_files = []; val_files = []
    for r in ROUTES:
        min_files = 10000
        total_files = {}
        ## Get filenames across specified magnifications ##
        for m in mags:
            curr_key = '%s_%s' %(r,m)

            curr_dir = '%s/%s/%s' %(rdir,r,m)
            curr_files = glob.glob(curr_dir + '/*')
            curr_files.sort()
            total_files[curr_key] = curr_files            
            if len(curr_files) < min_files:
                min_files = len(curr_files)

        curr_routes = []        
        for i in range(min_files):
            curr_val = total_files['%s_%s' %(r,mags[0])][i]
            for m in mags[1:]:
                curr_val += ' %s' %total_files['%s_%s' %(r,m)][i]
            curr_val += ' %s\n' %r
            curr_routes.append(curr_val)        

        num_per_img = 4
        ind_split = np.array_split(np.arange(len(curr_routes) // num_per_img), total_fold)
        
        ## Split into train/val based on specified fold ##
        curr_val_files = [ curr_routes[i] for i in range(ind_split[fold_num][0]*num_per_img,ind_split[fold_num][-1]*num_per_img) ]
        curr_train_files = [x for x in curr_routes if x not in curr_val_files]
        train_files += curr_train_files; val_files += curr_val_files

    random.shuffle(train_files)

    write_to_files(train_files, '%s/train_%s_fold%d.txt' %(trainval_file_dir,trainval_file_corename,fold_num))
    write_to_files(val_files, '%s/val_%s_fold%d.txt' %(trainval_file_dir,trainval_file_corename,fold_num))

def write_to_files(files, filename):
    '''
    Parse to text file
    '''

    f = open(filename,'w')
    for l in files:
        f.write(l)

    f.close()

def create_logger():
    ''' 
    Create logger object
    '''

    curr = datetime.datetime.now()
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    
    logger_name = './logs/%s_info.log'\
                    %(str(curr.year)+'_'+str(curr.month)+'_'\
                        +str(curr.day)+'_'+str(curr.hour)+\
                        str(curr.minute)+str(curr.second))
    
    logger = logging.getLogger(logger_name.split('/')[-1][:-4])
    logger.setLevel(logging.INFO)
    
    fh = logging.FileHandler(logger_name); fh.setLevel(logging.INFO)    
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)

    logger.addHandler(fh)
    
    return logger

def log_params(logger,params,datafile,mags,classes):
    '''
    Log user params
    '''

    logger.info('##### Params #####')
    logger.info('Loading data file: {:}'.format(datafile))
    logger.info('Network type: {:}'.format(params.type_network))         
    logger.info('Base network: {:}'.format(params.resnet))
    logger.info('Total number of epochs: {:}'.format(params.num_epochs))
    logger.info('Number of epochs for transfer learning training: {:}'.format(params.ep_init))
    logger.info('Batch size: {:}'.format(params.bz))
    logger.info('Learning rate: {:}'.format(params.lr))
    logger.info('Momentum: {:}\n'.format(params.momentum))

    logger.info('Magnifications: {:}'.format(mags))
    
    curr_classes = classes[0]
    for x in classes[1:]:
        curr_classes += ',%s' %x
    logger.info('Routes: {:}'.format(curr_classes))

    logger.info('##########\n')











 
