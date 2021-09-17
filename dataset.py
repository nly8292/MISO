from __future__ import print_function, division, absolute_import, unicode_literals
from PIL import Image as PILimage

from torch.utils.data import Dataset
from torchvision import transforms

ROUTES = ['U3O8ADU', 'U3O8AUC', 'U3O8MDU', 'U3O8SDU', 'UO2ADU', 'UO2AUCd', 'UO2AUCi', 'UO2SDU',\
            'UO3ADU', 'UO3AUC', 'UO3MDU', 'UO3SDU']

class SynRoutes(Dataset):
    def __init__(self, datafile, datatype):
        
        self.img_files, self.labels = self._parse_text_file(datafile)        
        self.num_mags = len(self.img_files) 
        self.transf = self._get_transforms(datatype)
        
    def __len__(self):
        return len(self.labels)  

    def __getitem__(self, idx):
        
        label = self.labels[idx]
        inp_tensor = {'label': label}
        
        for i in range(self.num_mags):
            curr_file = self.img_files['mag%d' %i][idx]
            curr_img = PILimage.open(curr_file).convert('RGB')
            inp_tensor['mag%d' %i] = self.transf(curr_img)
        
        return inp_tensor  

    def get_num_classes(self):
        '''
        Count the total number of classes
        '''

        return len(set(self.labels))

    def get_classes(self):
        '''
        Get classes' name
        '''

        uni_ind = set(self.labels)
        return [ROUTES[ui] for ui in uni_ind]

    def get_mags(self):
        '''
        Get all the magnifications used
        '''

        val = self.img_files['mag0'][0].split('/')[-2]
        for i in range(1,self.num_mags):
            val += ',%s' %self.img_files['mag%d'%i][0].split('/')[-2]
        return val


    def _parse_text_file(self, datafile):
        '''
        parse train/val text file into a dictionary input
        -> train/val text file template
            mag0/img0 mag1/img0 ... label
            mag0/img1 mag1/img1 ... label
            ...
            mag0/imgN mag1/imgN ... label
        '''

        f = open(datafile, 'r')
        img_files = {}; labels = [] 

        
        curr_val = f.readline().strip("\n").split(' ')
        for vi, v in enumerate(curr_val[:-1]):
            img_files['mag%d' %vi] = [v]                                   
            
        labels.append(ROUTES.index(curr_val[-1]))

        for line in f:
            try:
                curr_val = line.strip("\n").split(' ')              
            except ValueError: # Adhoc for test.
                print('Incompatible text format in data file!')                                   
                     
            ## Obtain file name for each magnification input ##
            for vi, v in enumerate(curr_val[:-1]):
                img_files['mag%d' %vi].append(v)                                   
                
            labels.append(ROUTES.index(curr_val[-1]))
        
        return img_files, labels

    def _get_transforms(self, datatype):
        '''
        Get data augmentation processes
        '''

        input_size = 224

        if datatype == 'train':
            return transforms.Compose([\
                    transforms.RandomHorizontalFlip(),\
                    transforms.RandomVerticalFlip(),\
                    transforms.ColorJitter(brightness=0.5),\
                    #transforms.ColorJitter(contrast=1.0),\
                    #transforms.RandomRotation(90.0),\
                    transforms.RandomResizedCrop(input_size),\
                    transforms.ToTensor(),\
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        else:
            return transforms.Compose([\
                    transforms.CenterCrop(input_size),\
                    transforms.ToTensor(),#]#,\
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        