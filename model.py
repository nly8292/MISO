from __future__ import print_function, division, absolute_import, unicode_literals
import time
import copy
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as cm
import os
import numpy as np

import torchvision
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter

import networks
from utils import *

ROUTES = ['U3O8ADU', 'U3O8AUC', 'U3O8MDU', 'U3O8SDU', 'UO2ADU', 'UO2AUCd', 'UO2AUCi', 'UO2SDU',\
            'UO3ADU', 'UO3AUC', 'UO3MDU', 'UO3SDU']

class Model(object):

	def __init__(self, params, num_classes, num_mags):
		
		self.params = params
		self.num_classes = num_classes
		self.num_mags = num_mags		

		self.criterion = nn.CrossEntropyLoss()		
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")		
			
		if self.params.type_network == 'unshared':
			self.model_ftrs = networks.Ftrs_UnSharedNet(self.num_mags, self.num_classes, self.params.resnet).to(self.device)		
		elif self.params.type_network == 'shared':
			self.model_ftrs = networks.Ftrs_SharedNet(self.num_mags, self.num_classes, self.params.resnet).to(self.device)
		self.model_clf = networks.MLP_Clf(self.num_mags, self.num_classes, self.params.resnet).to(self.device)		

		params_ftrs, params_fcbn_ftrs = self._get_network_params(self.model_ftrs)
		params_clf, params_fcbn_clf = self._get_network_params(self.model_clf)

		params_net = params_ftrs + params_clf
		params_fcbn = params_fcbn_ftrs + params_fcbn_clf
				
		self.optimizer1 = optim.SGD([{'params':params_fcbn, 'lr':params.lr*10}], momentum=0.9)#momentum)
		self.optimizer2 = optim.SGD([{'params': params_net, 'lr':params.lr},
									 {'params': params_fcbn, 'lr':params.lr}], momentum=0.9)#momentum)

		self.saved_exp_name = '%s_fold%d_%s_%s' %(params.datafile,params.fold_num,params.resnet,params.type_network)
		if not os.path.exists('./saved_models/'):
			os.makedirs('./saved_models/')
		if not os.path.exists('./tensorboards/'):
			os.makedirs('./tensorboards/')
		if not params.isTesting:
			self.curr_writer = SummaryWriter('./tensorboards/%s' %self.saved_exp_name)
			if not os.path.exists('./saved_models/%s' %self.saved_exp_name):
				os.makedirs('./saved_models/%s' %self.saved_exp_name)
	
	def _get_network_params(self, net):
		'''
		Split up network params
		'''

		params = []; params_fcbn = []
		for name, param in net.named_parameters():			
			if 'fc' in name or 'BN' in name:
				params_fcbn.append(param)
			else:
				params.append(param)
		return params, params_fcbn	

	def train_model(self, dataloaders, logger):

		since = time.time()
		
		best_acc = 0.0; best_ep = -1
		num_epochs = self.params.num_epochs

		for epoch in range(num_epochs):
			print('-' * 10)
			print('Epoch {}/{}'.format(epoch, num_epochs - 1))
			
			trainingPhase = ['train','val']			
			for phase in trainingPhase:				
				if phase == 'train':       
					self.model_ftrs.train()
					self.model_clf.train()
				else:
					self.model_ftrs.eval()
					self.model_clf.eval()

				running_loss = 0.0; running_corrects = 0
				
				for inp_tensor in dataloaders[phase]:	

					labels = inp_tensor['label'].to(self.device)					
					## Parse input tensors into a list
					inp = [inp_tensor['mag0'].to(self.device)]
					for i in range(1,len(inp_tensor)-1):						
						inp.append(inp_tensor['mag%d' %i].to(self.device))				
										
					if epoch < self.params.ep_init:
						self.optimizer1.zero_grad()
					else:
						self.optimizer2.zero_grad()
                              
					with torch.set_grad_enabled(phase == 'train'):																					
						ftrs = self.model_ftrs(inp)								
						outputs = self.model_clf(ftrs)																				
						loss = self.criterion(outputs, labels)

						_, preds = torch.max(outputs, 1)						

						if phase == 'train':														
							loss.backward()							
							if epoch < self.params.ep_init:								
								self.optimizer1.step()
							else:
								self.optimizer2.step()

					running_loss += loss.item() * inp[-1].size(0)
					running_corrects += torch.sum(preds == labels.data)					
				
				epoch_loss = running_loss / len(dataloaders[phase].dataset)				
				epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset) 
				
				if epoch < self.params.ep_init :
					curr_lr = self.optimizer1.param_groups[0]['lr']
				else:
					curr_lr = self.optimizer2.param_groups[0]['lr']

				print('{:} Loss: {:.4f} Acc: {:.4f} Lr: {:.6f}'.format(phase, epoch_loss, epoch_acc, curr_lr))

				self.curr_writer.add_scalar('%s/Loss' %phase, epoch_loss, epoch)   
				self.curr_writer.add_scalar('%s/Acc' %phase, epoch_acc, epoch)
				
				curr_info = '{:} Phase -> Epoch#{:}: Loss - {:.4f}; Acc:{:.4f}'.format(phase,epoch,epoch_loss,epoch_acc)
				if phase == 'val':									
					curr_info += '\n'
					if epoch_acc >= best_acc:
						best_acc = epoch_acc
						best_ep = epoch						
						best_model_ftrs = self._saved_models(self.model_ftrs, '%s_ftrs' %self.saved_exp_name)
						best_model_clf = self._saved_models(self.model_clf, '%s_clf' %self.saved_exp_name)

				logger.info(curr_info)
								
		time_elapsed = time.time() - since
		print('\nTraining completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
		logger.info('Best epoch saved: {:}'.format(best_ep))
	
		self.model_ftrs.load_state_dict(best_model_ftrs)
		self.model_clf.load_state_dict(best_model_clf)

		self.curr_writer.close()
				

	def predict(self, dataloader, classes):
	
		self.model_ftrs.eval()		
		self.model_clf.eval()

		acc = 0.0
		total_labels = torch.ones((1,), dtype = torch.int64).to(self.device)
		total_preds = torch.ones((1,), dtype = torch.int64).to(self.device)		
				
		for inp_tensor in dataloader:		
			labels = inp_tensor['label'].to(self.device)			
			inp = [inp_tensor['mag0'].to(self.device)]
			for i in range(1,len(inp_tensor)-1):						
				inp.append(inp_tensor['mag%d' %i].to(self.device))					
			
			with torch.set_grad_enabled(False):					
				ftrs = self.model_ftrs(inp)								
				outputs = self.model_clf(ftrs)				
				
				_, preds = torch.max(outputs, 1)
				acc += torch.sum(preds == labels.data)				
				
				total_labels = torch.cat((total_labels, labels), 0)				
				total_preds = torch.cat((total_preds, preds), 0)
		
		acc = float(acc.item()) / len(dataloader.dataset)
		total_labels = total_labels[1:].cpu().numpy()
		total_preds = total_preds[1:].cpu().numpy()

		maj_vec, maj_labels = majority_vote(total_preds, total_labels)
		maj_conf_matrix = cm(maj_labels, maj_vec, labels=np.arange(len(classes)))
		plot_confusion_matrix(maj_conf_matrix, classes, self.saved_exp_name+'_maj', normalize=True)
			
		print('Validation Accuracy = {:.6f}'.format(acc))
		
		conf_matrix = cm(total_labels, total_preds, labels=np.arange(len(classes)))		
		plot_confusion_matrix(conf_matrix, classes, self.saved_exp_name, normalize=True)

	
	def load_trained_model(self, saved_model_dir):	
		'''
		Load pretrained model
		'''

		self.model_ftrs.load_state_dict(torch.load('%s/%s/%s_ftrs.pt' %(saved_model_dir,self.saved_exp_name,self.saved_exp_name)))
		self.model_clf.load_state_dict(torch.load('%s/%s/%s_clf.pt' %(saved_model_dir,self.saved_exp_name,self.saved_exp_name)))

	def _saved_models(self, model, model_name):
		'''
		Save model weights
		'''

		best_model = copy.deepcopy(model.state_dict())
		torch.save(model.state_dict(), './saved_models/%s/%s.pt' %(self.saved_exp_name,model_name))

		return best_model


