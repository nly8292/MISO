import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class MLP_Clf(nn.Module):
    '''
    Classification layers
    '''

    def __init__(self, num_mags, num_classes, resnet):
        super(MLP_Clf, self).__init__()

        if resnet == 'resnet18':
            num_ftrs = 512*2            
        elif resnet == 'resnet34':
            num_ftrs = 512*2            
        elif resnet == 'resnet50':
            num_ftrs = 2048*2

        fc_in = (num_mags * num_ftrs) // 4        
        self.BN1 = torch.nn.BatchNorm1d(fc_in)        
        self.fc_out = nn.Linear(fc_in, num_classes)        
        self.do = torch.nn.Dropout(p=0.5)

    def forward(self, ftrs):
        out =  self.fc_out( self.do( self.BN1( ftrs ) ) ) 
        return out

class Ftrs_SharedNet(nn.Module):
    '''
    Feature Extraction with shared network
    '''

    def __init__(self, num_mags, num_classes, resnet):
        super(Ftrs_SharedNet, self).__init__()                   
        
        if resnet == 'resnet18':
            num_ftrs = 512*2
            curr_model = models.resnet18(pretrained=True)
        elif resnet == 'resnet34':
            num_ftrs = 512*2
            curr_model = models.resnet34(pretrained=True)
        elif resnet == 'resnet50':
            num_ftrs = 2048*2
            curr_model = models.resnet50(pretrained=True)
        self.model_class = nn.Sequential(*list(curr_model.children())[:-2])

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.pool_bn = torch.nn.BatchNorm1d(num_ftrs)        

        self.fc1 = nn.Linear(num_ftrs, num_ftrs // 4)                            

    def forward(self, x):

        ftrs = []
        for i, xi in enumerate(x):
            resnet_out = self.model_class(xi)
            avg_pool = self.avg_pool(resnet_out)
            max_pool = self.max_pool(resnet_out)
            curr_pool = self.pool_bn(torch.squeeze(torch.cat((avg_pool,max_pool),1)))
            
            curr_ftrs = F.relu( (self.fc1( curr_pool ) ))
            ftrs.append(curr_ftrs)
        ftrs = torch.cat(ftrs, 1)
        
        return ftrs


class Ftrs_UnSharedNet(nn.Module):
    '''
    Feature Extraction with multiple networks
    '''

    def __init__(self, num_mags, num_classes, resnet):

        super(Ftrs_UnSharedNet, self).__init__()                     

        self.nets = nn.ModuleList([Ftrs_SharedNet(1, num_classes, resnet) for i in range(num_mags)])
           
    def forward(self, x):  
        ftrs = []
        for xi, curr_net in zip(x, self.nets):
            
            curr_ftrs = curr_net([xi])            
            ftrs.append(curr_ftrs)
        ftrs = torch.cat(ftrs, 1)

        return ftrs

