# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 22:05:39 2021

@author: Ding
"""

import os
import time
import shutil
import argparse
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score
import resnet3d
import torch.backends.cudnn as cudnn
from sklearn.model_selection import StratifiedKFold
from torch.utils.data.dataset import Dataset
import torch.multiprocessing as mp
from sklearn.datasets import load_digits
import torchvision.models as models
#this part is the general setting of the framwork
parser = argparse.ArgumentParser(description='basic traning python file')

parser.add_argument('--epochs', default=150, type=int, 
                    help='number of total epochs to run')

parser.add_argument('--batch_size',default=128, type=int,
                    help='batch size ')

parser.add_argument('--lr',default=1e-3,type=float,
                    help='initial learning rate')

parser.add_argument('--momentum', default=0.9, type=float, 
                    help='momentum')

parser.add_argument('--weight_decay',
                    default=1e-4,
                    type=float,
                    help='weight decay (default: 1e-4)')

parser.add_argument('-p', '--print-freq', default=10, type=int, 
                    help='print frequency (default: 10)')

parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', 
                    help='evaluate model on validation set')

parser.add_argument('--pretrained', dest='pretrained', action='store_true', 
                    help='use pre-trained model')

parser.add_argument('--seed', default=None, type=int, 
                    help='seed for initializing training. ')

parser.add_argument('-a','--arch',default='resnet18',
                    help='model name')

#define the model structure

class Basic_Model(nn.Module):
    def __init__(self):
        super(Basic_Model,self).__init__()
        
        self.layers=nn.Sequential(nn.Linear(in_features=64, out_features=16),
                                  nn.ReLU(),
                                  nn.Linear(in_features=16, out_features=10)
                )
        
    def forward(self,x):
        output=self.layers(x)
        softmax_result = nn.functional.softmax(output, dim=1)
        
        return softmax_result
def main():
    #load the general setting
    args = parser.parse_args()
    #data set loading
    X,y = load_digits(return_X_y=True)
    #using 5-cross validation
    kf = StratifiedKFold(n_splits=5,shuffle=True,random_state=2021)
    #load the model
    model=Basic_Model()
    
    original_params=model.state_dict()
    n_split=0
    result_mat=np.zeros([5,7])
    
    for train_idx, test_idx in kf.split(X,y):
        model.load_state_dict(original_params)
        if torch.cuda.is_available():
            model.cuda() 
            criterion = nn.CrossEntropyLoss().cuda()
            
        else:
            criterion = nn.CrossEntropyLoss().cpu()
        best_acc1= .0
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        
        train_data=X[train_idx,:]
        train_label=y[train_idx]
        train_data=torch.from_numpy(train_data)
        train_label=torch.from_numpy(train_label)
        
        test_data=X[test_idx,:]
        test_label=y[test_idx]
        test_data=torch.from_numpy(test_data)
        test_label=torch.from_numpy(test_label)
        
        train_dataset = torch.utils.data.TensorDataset(train_data,train_label)
        train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                                   batch_size=64,
                                                   shuffle=True,
                                                   num_workers=2)

        test_dataset = torch.utils.data.TensorDataset(test_data,test_label)
        test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                                   batch_size=64,
                                                   shuffle=False,
                                                   num_workers=2)  
        
        for epoch in range(args.epochs):
            
            adjust_learning_rate(optimizer, epoch, args)
            
            train(train_loader, model, criterion, optimizer, epoch, args)
            
            
            

def train(train_loader, model, criterion, optimizer, epoch, args):
    
    model.train()
    
    total_label=[]
    total_predict_label=[]
    
    for i,(data,label) in enumerate(train_loader):
        
        if torch.cuda.is_available():
            data=data.cuda()
            label=label.cuda()
        
        output=model(data)
        
        predict_label = torch.argmax(output, dim=1)
        
        loss = criterion(output, label)
        total_label.append(label)    
        total_predict_label.append(predict_label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    
        
        
        
        

            
        
    
    
    
    
    
    
    
    
    
    