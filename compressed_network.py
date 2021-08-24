# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 18:09:58 2021

@author: Ding
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 17:27:12 2021

@author: Ding
"""

#main.py
import os
import time
import shutil
import argparse
import torch
import torch.nn as nn
import numpy as np
import resnet3d
import torch.backends.cudnn as cudnn
from sklearn.model_selection import StratifiedKFold
from torch.utils.data.dataset import Dataset
import torch.multiprocessing as mp
#settings
parser = argparse.ArgumentParser(description='single brain region model')

parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--epochs', default=50, type=int, 
                    help='number of total epochs to run')
parser.add_argument('--batch_size',default=128, type=int,
                    help='batch size of all GPUs')
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



def reduced_mean(tensor, nprocs):
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= nprocs
    return rt

def gather_tensor(tensor):
    rt=[tensor.clone() for _ in range(torch.distributed.get_world_size()) ]
    torch.distributed.all_gather(rt, tensor)
    concat= torch.cat(rt,dim=0)
    return concat

class InputDataset(Dataset):
    def __init__(self,inputdata,inputlabel):
        self.Data=inputdata
        self.Label=inputlabel
 
    def __getitem__(self, index):
        data=self.Data[index]
        label=self.Label[index]
        return data, label 
 
    def __len__(self):
        return len(self.Data)
    
def kronecker_product(feature_set,args):
#    n_sample, size1= x1.shape
#    n_sample, size2 =x2.shape
#    n_sample, size3 =x3.shape
    kron_prod=[]
    for idx in range(args.n_data):
        exec('n_sample,size'+str(idx)+'=feature_set['+str(idx)+']')
        exec('feature'+str(idx)+'=feature_set['+str(idx)+']')
    for i in range (n_sample):
#        current_feature1=x1[i,:]
#        current_feature2=x2[i,:]
#        current_feature3=x3[i,:]
        for idx in range(args.n_data):
            exec('current_feature'+str(idx)+'=feature'+str(idx)+'['+str(i)+',:]')
            if idx==0:
                current_feature0=current_feature0.reshape(1,size0)
            else:
                exec('current_feature'+str(idx)+'=current_feature'+str(idx)+'.reshape(size'+str(idx)+',1)')
#        current_feature1=current_feature1.reshape(1, size1)
#        current_feature2=current_feature2.reshape(size2, 1)
#        current_feature3=current_feature3.reshape(size3, 1)
        temp_feature=(current_feature0*current_feature1).view(1,size0*size1)
        for idx in range(1,args.n_data):
            n_data,current_size=temp_feature.shape
            exec('current_feature=(temp_feature*current_feature'+str(idx)+').view(1,n_size*size'+str(idx)+')')
        current_feature=current_feature.view(-1)

        kron_prod.append(current_feature)
    kron_prod=torch.cat(kron_prod, dim=0)
    
    output=kron_prod.view(n_sample,(args.feature_size)**(args.n_data))
    return output
    


def feature_kron(feature_set,args):
#    assert feature1.shape[0] == feature2.shape[0]==feature3.shape[0]
    feature_kron=kronecker_product(feature_set,args)
#    feature_kron.unsqueeze(dim=2).unsqueeze(dim=3).unsqueeze(dim=4)
    return feature_kron
    
def basic_combination(model,transfered_size,feature_size=128):
#    x=original_size[0]/transfered_size[0]
#    y=original_size[1]/transfered_size[1]
#    z=original_size[2]/transfered_size[2]
    basic_combination=nn.Sequential(nn.Upsample(size=transfered_size),
                                 model,
                                 nn.Flatten(),
                                 nn.Linear(512,feature_size),
                                 nn.Linear(feature_size,2))
    return basic_combination
    
    
    
class Main_Net(nn.Module):
    def __init__(self,feature_extraction_layer,args):
        super(Main_Net,self).__init__()
        
        self.feature_extraction_layer=feature_extraction_layer
#        self.branch1=branch1
#        self.branch2=branch2
#        self.branch3=branch3
        for idx in range(args.n_data):
            exec('self.branch'+str(idx)+'=args.fc'+str(idx))
        
        self.fcn=nn.Sequential(nn.Linear(in_features=(args.feature_size)**(args.n_data), out_features=2, bias=True)
                                )
        
    def forward(self,x, args):
#        x1=self.feature_extraction_layer(x1)
#        x2=self.feature_extraction_layer(x2)
#        x3=self.feature_extraction_layer(x3)
        feature_set=[]
        for idx in range(args.n_data):
            exec('x'+str(idx)+'=self.feature_extraction_layer(x.x'+str(idx)+')')
            exec('feature'+str(idx)+'=self.branch'+str(idx)+'(x'+str(idx)+')')
            exec('feature_set.append(feature'+str(idx)+')')
#        feature1=self.branch1(x1)
#        feature2=self.branch2(x2)
#        feature3=self.branch3(x3)
        
#        feature1=feature1.view(feature1.size(0),-1)
#        feature2=feature2.view(feature2.size(0),-1)
#        feature3=feature3.view(feature3.size(0),-1)
        


        feature=feature_kron(feature_set,args)
        output=self.fcn1(feature)

        
        return output
        
class Compressed_Net(nn.Module):
    def __init__(self, input_size,compressed_size):
        super(Compressed_Net,self).__init__()
        self.input_size=input_size
        self.compressed_size=compressed_size
        self.compress_layer=nn.Linear(in_features=input_size, out_features=compressed_size, bias=True)
    def forward(self,x):
        n_sample=x.shape[0]
        x=x.view(n_sample,-1)
        output=self.compress_layer(x)
        
        return output
        
class Uncompressed_Net(nn.Module):
    def __init__(self, input_size,uncompressed_size):
        super(Uncompressed_Net,self).__init__()
        self.input_size=input_size
        self.uncompressed_size=uncompressed_size
        self.uncompress_layer=nn.Linear(in_features=input_size, out_features=uncompressed_size, bias=True)
    def forward(self,x):

        output=self.uncompress_layer(x)
        
        return output

class basic_model(nn.Module):
    def __init__(self, model,rank):
        super(basic_model,self).__init__()
        self.feature_layer=nn.Sequential(*list(model.children())[:-1])
        self.flatten=nn.Flatten()
        self.fcn1=nn.Linear(512,rank,bias=True)
        self.fcn2=nn.Linear(rank,2)
    def forward(self,x):

        x=self.feature_layer(x)
        x=self.flatten(x)
        x=self.fcn1(x)
        output=self.fcn2(x)
        return output
    
#class feature_extract(nn.Module):
#    def __init__(self, model, rank):
#        super(feature_extract,self).__init__()
#        self.feature_layer=nn.Sequential(*list(model.children())[:-1])
#        self.fcn1=nn.Linear(512,rank,bias=True)
#
#    def forward(self,x):
#
#        x=self.feature_layer(x)
#        x=x.view(x.size(0),-1)
#        output=self.fcn1(x)
#        return output

def main():
    args = parser.parse_args()
    
#    args.nprocs = 2
    args.nprocs = torch.cuda.device_count()
    
#    os.environ['CUDA_VISIBLE_DEVICES']='4,5'
    os.environ['MASTER_ADDR'] ='124.16.75.175'
    os.environ['MASTER_PORT'] = '8888'
    
    
#    data_path=['/home/ding/exp_2/data/Hippocampus_LAD_NC.npy',
#               '/home/ding/exp_2/data/Hippocampus_R_data.npy',
#               '/home/ding/exp_2/data/ParaHippocampus_L_data.npy']
    data_path=["D:\\temp_python\\Hippocampus_LAD_NC.npy",
               "D:\\temp_python\\Hippocampus_R_data.npy"]
    n_data=0
    for path in data_path:
        
        exec('args.data'+str(n_data)+'=np.load(r\''+path+'\')')
        n_data=n_data+1
    
    
#    data_path1='/home/ding/exp_2/data/Hippocampus_LAD_NC.npy'
#    data_path2='/home/ding/exp_2/data/Hippocampus_R_data.npy'
#    data_path3='/home/ding/exp_2/data/ParaHippocampus_L_data.npy'
#
#    data1=np.load(data_path1)
#    data2=np.load(data_path2) 
#    data3=np.load(data_path3)
    
    
#    label=np.concatenate((np.ones([200,1],dtype=float),np.zeros([235,1],dtype=float)),0)
    
    args.label=np.concatenate((np.ones([200,1],dtype=float),np.zeros([235,1],dtype=float)),0)
    
#    args.data1=data1
#    args.data2=data2
#    args.data3=data3    
    
    model=resnet3d.ResNet(resnet3d.BasicBlock, [1, 1, 1, 1],resnet3d.get_inplanes(),
                     n_input_channels=1,
                     conv1_t_size=27,
                     conv1_t_size2=27,
                     conv1_t_size3=20,
                     conv1_t_stride=1,
                     no_max_pool=False,
                     shortcut_type='B',
                     n_classes=2)
    
    model=nn.Sequential(*list(model.children())[:-1])
    args.feature_size=128
    exp_type='tri_kp_common_with_fixed_fcn_'
    
    save_path=['/home/ding/exp_2/result/result_Hippocampus_L_',
               '/home/ding/exp_2/result/result_Hippocampus_R_',
               '/home/ding/exp_2/result/result_ParaHippocampus_L_']
    n_save_path=0
    for path in save_path:
        
        exec('args.save_path'+str(n_save_path)+'='+path+exp_type)
        n_save_path=n_save_path+1
#    args.save_path1='/home/ding/exp_2/result/result_Hippocampus_L_'+exp_type
#    args.save_path2='/home/ding/exp_2/result/result_Hippocampus_R_'+exp_type
#    args.save_path3='/home/ding/exp_2/result/result_ParaHippocampus_L_'+exp_type
    
#    args.orginal_params_save_path1='/home/ding/exp_2/data/Hippocampus_L_original_'+exp_type
#    args.orginal_params_save_path2='/home/ding/exp_2/data/Hippocampus_R_original_'+exp_type
#    args.orginal_params_save_path3='/home/ding/exp_2/data/ParaHippocampus_L_original_'+exp_type
    
#    args.params_save_path1='/home/ding/exp_2/data/Hippocampus_L_kronecker_'+exp_type
#    args.params_save_path2='/home/ding/exp_2/data/Hippocampus_R_kronecker_'+exp_type
#    args.params_save_path3='/home/ding/exp_2/data/ParaHippocampus_L_kronecker_'+exp_type
   
    args.model=model

    
    args.save_path_main='/home/ding/exp_2/result/result_'+exp_type
    
    args.net_module_flag=-1
    args.n_data=n_data
    mp.spawn(main_worker, nprocs=args.nprocs, args=(args.nprocs, args))
    
def main_worker(local_rank,nprocs, args):
    args.local_rank = local_rank
    torch.distributed.init_process_group(backend='nccl', world_size=args.nprocs, rank=local_rank)
    torch.cuda.set_device(local_rank)
    cudnn.benchmark = True
    args.batch_size = int(args.batch_size / args.nprocs)        
    
    kf = StratifiedKFold(n_splits=5,shuffle=True,random_state=2021)
    feature_extraction_layer=args.model
    
    set_gradient(feature_extraction_layer,n_layers=36,state=True)
    original_params=feature_extraction_layer.state_dict()

    
    n_split=0
    for idx in range(args.n_data):
        exec('result_mat'+str(idx)+'=np.zeros([5,7])')
#    result_mat1=np.zeros([5,7])
#    result_mat2=np.zeros([5,7])
#    result_mat3=np.zeros([5,7])
    result_mat_main=np.zeros([5,7])
    for train_idx, test_idx in kf.split(args.data1,args.label):
        feature_extraction_layer.load_state_dict(original_params)
        for idx in range(args.n_data):
            args.net_module_flag=-1
            exec('current_model=basic_combination(feature_extraction_layer,args.data'+str(idx)+'.shape[1:], args.feature_size)')
#            current_model=basic_combination(feature_extraction_layer,args.data1.shape[1:])
            
            current_model.cuda(local_rank)
            model_para = torch.nn.parallel.DistributedDataParallel(current_model, device_ids=[local_rank])
            exec('best_acc'+str(idx)+'= .0')
#            best_acc1= .0
            criterion = nn.CrossEntropyLoss().cuda(local_rank)
            optimizer = torch.optim.SGD(model_para.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            exec('train_loader,test_loader,train_sampler,test_sampler= dataloader_preprocessing(train_idx,test_idx,args,args.data'+str(idx)+') ')
#            train_loader,test_loader,train_sampler,test_sampler= dataloader_preprocessing(train_idx,test_idx,args,args.data1)     

            for epoch in range(args.epochs):
            
                train_sampler.set_epoch(epoch)
                test_sampler.set_epoch(epoch)
            
                adjust_learning_rate(optimizer, epoch, args)
            
                train(train_loader, model_para, criterion, optimizer, epoch, local_rank, args)
            
#                acc1 = validate(test_loader, model_para, criterion, local_rank, args)
                exec('acc'+str(idx)+'= validate(test_loader, model_para, criterion, local_rank, args)')
#                is_best = acc1.Accuracy > best_acc1
                exec('is_best = acc'+str(idx)+'.Accuracy > best_acc'+str(idx))
#                best_acc1 = max(acc1.Accuracy, best_acc1)

#                if args.local_rank == 0:
#                    save_checkpoint(
#                        {
#                            'epoch': epoch + 1,
#                            'arch': args.arch,
#                            'state_dict': model_para.module.state_dict(),
#                            'best_acc1': best_acc1,
#                        }, is_best)
                exec('result_path=args.save_path'+str(idx)+'+str(n_split)+\'time.npy\'')
#                result_path=args.save_path1+str(n_split)+'time.npy'
                exec('result_mat'+str(idx)+'[n_split,0]=acc'+str(idx)+'.Accuracy')
#                result_mat1[n_split,0]=acc1.Accuracy
                exec('result_mat'+str(idx)+'[n_split,1]=acc'+str(idx)+'.TPR')
#                result_mat1[n_split,1]=acc1.TPR
                exec('result_mat'+str(idx)+'[n_split,2]=acc'+str(idx)+'.FPR')
#                result_mat1[n_split,2]=acc1.FPR
                exec('result_mat'+str(idx)+'[n_split,3]=acc'+str(idx)+'.Precision')
#                result_mat1[n_split,3]=acc1.Precision
                exec('result_mat'+str(idx)+'[n_split,4]=acc'+str(idx)+'.F1score')
#                result_mat1[n_split,4]=acc1.F1score
                exec('result_mat'+str(idx)+'[n_split,5]=acc'+str(idx)+'.Kappa')                
#                result_mat1[n_split,5]=acc1.Kappa
            exec('np.save(result_path,result_mat'+str(idx)+')')
#            np.save(result_path,result_mat1)
            exec('fc'+str(idx)+'=nn.Sequential(*list(current_model.children())[-3:-1])')
        
#            fc1=nn.Sequential(*list(current_model.children())[-3:-1])
#            set_gradient(fc1,n_layers=2,state=False)

            exec('set_gradient(fc'+str(idx)+',n_layers=2,state=False)')
            exec('args.fc'+str(idx)+'=fc'+str(idx))
            if idx==0:
                set_gradient(feature_extraction_layer,n_layers=36,state=False)
        
#        current_model=basic_combination(feature_extraction_layer,args.data2.shape[1:])
#
#        current_model.cuda(local_rank)
#        model_para = torch.nn.parallel.DistributedDataParallel(current_model, device_ids=[local_rank])
#        best_acc2= .0
#        criterion = nn.CrossEntropyLoss().cuda(local_rank)
#        optimizer = torch.optim.SGD(model_para.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
#
#        train_loader,test_loader,train_sampler,test_sampler= dataloader_preprocessing(train_idx,test_idx,args,args.data2)     
#
#        for epoch in range(args.epochs):
#            
#            train_sampler.set_epoch(epoch)
#            test_sampler.set_epoch(epoch)
#            
#            adjust_learning_rate(optimizer, epoch, args)
#            
#            train(train_loader, model_para, criterion, optimizer, epoch, local_rank, args)
#            
#            acc2 = validate(test_loader, model_para, criterion, local_rank, args)
#            
#            is_best = acc2.Accuracy > best_acc2
#            best_acc2 = max(acc2.Accuracy, best_acc2)
#
#            if args.local_rank == 0:
#                save_checkpoint(
#                    {
#                        'epoch': epoch + 1,
#                        'arch': args.arch,
#                        'state_dict': model_para.module.state_dict(),
#                        'best_acc1': best_acc2,
#                    }, is_best)
#        
#        result_path=args.save_path2+str(n_split)+'time.npy'
#        result_mat2[n_split,0]=acc2.Accuracy
#        result_mat2[n_split,1]=acc2.TPR
#        result_mat2[n_split,2]=acc2.FPR
#        result_mat2[n_split,3]=acc2.Precision
#        result_mat2[n_split,4]=acc2.F1score
#        result_mat2[n_split,5]=acc2.Kappa
#        
#        np.save(result_path,result_mat2)
#
#        fc2=nn.Sequential(*list(current_model.children())[-3:-1])
#        set_gradient(fc2,n_layers=2,state=False)
#        current_model=basic_combination(feature_extraction_layer,args.data3.shape[1:])
#
#
#        current_model.cuda(local_rank)
#        model_para = torch.nn.parallel.DistributedDataParallel(current_model, device_ids=[local_rank])
#        best_acc3= .0
#        criterion = nn.CrossEntropyLoss().cuda(local_rank)
#        optimizer = torch.optim.SGD(model_para.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
#
#        train_loader,test_loader,train_sampler,test_sampler= dataloader_preprocessing(train_idx,test_idx,args,args.data3)     
#
#        for epoch in range(args.epochs):
#            
#            train_sampler.set_epoch(epoch)
#            test_sampler.set_epoch(epoch)
#            
#            adjust_learning_rate(optimizer, epoch, args)
#            
#            train(train_loader, model_para, criterion, optimizer, epoch, local_rank, args)
#            
#            acc3 = validate(test_loader, model_para, criterion, local_rank, args)
#            
#            is_best = acc3.Accuracy > best_acc3
#            best_acc3 = max(acc3.Accuracy, best_acc3)
#
#            if args.local_rank == 0:
#                save_checkpoint(
#                    {
#                        'epoch': epoch + 1,
#                        'arch': args.arch,
#                        'state_dict': model_para.module.state_dict(),
#                        'best_acc1': best_acc3,
#                    }, is_best)
#        
#        result_path=args.save_path3+str(n_split)+'time.npy'
#        result_mat3[n_split,0]=acc3.Accuracy
#        result_mat3[n_split,1]=acc3.TPR
#        result_mat3[n_split,2]=acc3.FPR
#        result_mat3[n_split,3]=acc3.Precision
#        result_mat3[n_split,4]=acc3.F1score
#        result_mat3[n_split,5]=acc3.Kappa
#        
#        np.save(result_path,result_mat3)
#        fc3=nn.Sequential(*list(current_model.children())[-3:-1])
#        set_gradient(fc3,n_layers=2,state=False)



        current_model=Main_Net(feature_extraction_layer, args)
        
        
        current_model.cuda(local_rank)
        model_para = torch.nn.parallel.DistributedDataParallel(current_model, device_ids=[local_rank],find_unused_parameters=True)
        best_acc_main= .0
        criterion = nn.CrossEntropyLoss().cuda(local_rank)
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model_para.parameters()), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        flatten_data=data_synthesize(args)
        train_loader,test_loader,train_sampler,test_sampler= dataloader_preprocessing(train_idx,test_idx,args,flatten_data)     
        args.net_module_flag=1
        
        for epoch in range(args.epochs):
            
            train_sampler.set_epoch(epoch)
            test_sampler.set_epoch(epoch)
            
            adjust_learning_rate(optimizer, epoch, args)
            
            train(train_loader, model_para, criterion, optimizer, epoch, local_rank, args)
            
            acc_main = validate(test_loader, model_para, criterion, local_rank, args)
            
            is_best = acc_main.Accuracy > best_acc_main
            best_acc_main= max(acc_main.Accuracy, best_acc_main)

            if args.local_rank == 0:
                save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model_para.module.state_dict(),
                        'best_acc1': best_acc_main,
                    }, is_best)
        
        result_path=args.save_path_main+str(n_split)+'time.npy'
        result_mat_main[n_split,0]=acc_main.Accuracy
        result_mat_main[n_split,1]=acc_main.TPR
        result_mat_main[n_split,2]=acc_main.FPR
        result_mat_main[n_split,3]=acc_main.Precision
        result_mat_main[n_split,4]=acc_main.F1score
        result_mat_main[n_split,5]=acc_main.Kappa
        np.save(result_path,result_mat_main)
        
        n_split=n_split+1
        
        
def set_gradient(model,n_layers=38, state=False):
        params = list(model.named_parameters())
        layer_list=[]

        for i in range(n_layers):
            layer_list.append(params[i][0])
        
        for k,v in model.named_parameters():
            if k not in layer_list:
                v.requires_grad=state

               
                
def data_synthesize(args):
#    n_sample=data1.shape[0]
#    x1=data1.shape[1]
#    y1=data1.shape[2]
#    z1=data1.shape[3]
#    
#    data1_flatten=data1.reshape(n_sample,x1*y1*z1)
#    x2=data2.shape[1]
#    y2=data2.shape[2]
#    z2=data2.shape[3]
#    data2_flatten=data2.reshape(n_sample,x2*y2*z2)
#    
#    
#    x3=data3.shape[1]
#    y3=data3.shape[2]
#    z3=data3.shape[3]
#    data3_flatten=data3.reshape(n_sample,x3*y3*z3)
    concatenate_command_line='data=np.concatenate(('
    for idx in range(args.n_data):
        exec('data'+str(idx)+'=args.data'+str(idx))
        exec('n_sample=data'+str(idx)+'.shape[0]')
        exec('x'+str(idx)+'=data'+str(idx)+'.shape[1]')
        exec('y'+str(idx)+'=data'+str(idx)+'.shape[2]')
        exec('z'+str(idx)+'=data'+str(idx)+'.shape[3]')
        exec('data'+str(idx)+'_flatten=data'+str(idx)+'.reshape(n_sample,x'+str(idx)+'*y'+str(idx)+'*z'+str(idx)+')')
        concatenate_command_line=concatenate_command_line+'data'+str(idx)+'_flatten,'
    concatenate_command_line=concatenate_command_line[:-1]+'),axis=1)'
    exec(concatenate_command_line)
#    data=np.concatenate((data1_flatten,data2_flatten,data3_flatten),axis=1)
    return data
    
    
def dataloader_preprocessing(train_idx,test_idx,args,data):
        train_data=data[train_idx]
        train_label=args.label[train_idx]
        train_data=np.array(train_data)
        train_label=np.array(train_label)
        train_data=torch.from_numpy(train_data)
        train_label=torch.from_numpy(train_label)
        
        
        test_data=data[test_idx]
        test_label=args.label[test_idx]
        test_data=np.array(test_data)
        test_label=np.array(test_label)
        test_data=torch.from_numpy(test_data)
        test_label=torch.from_numpy(test_label)
        
        train_dataset = torch.utils.data.TensorDataset(train_data,train_label)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=2,
                                               pin_memory=True,
                                               sampler=train_sampler)
        
        
        test_dataset = torch.utils.data.TensorDataset(test_data,test_label)
        

        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=2,
                                               pin_memory=True,
                                               sampler=test_sampler)
        
        return train_loader, test_loader,train_sampler,test_sampler
        
def train_compressed(train_loader, model,input_model,criterion, optimizer, epoch, local_rank, args):        
    model.train()    

    for i,(data,label) in enumerate(train_loader):
        
        data=data.unsqueeze(dim=1)
        label=label.squeeze()
        data = data.type(torch.FloatTensor)
        label = label.type(torch.LongTensor)
        data=data.cuda(local_rank,non_blocking=True)
        label=label.cuda(local_rank,non_blocking=True)
        
        
        output_original=input_model(data)   
        output=model(output_original)
        output=torch.unsqueeze(output,dim=2)
        output=torch.unsqueeze(output,dim=3)
        output=torch.unsqueeze(output,dim=4)

        loss = criterion(output_original, output)
        
        torch.distributed.barrier()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
def train(train_loader, model, criterion, optimizer, epoch, local_rank, args):
    model.train()
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    Accuracy=AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(train_loader), [batch_time, losses, Accuracy],
                             prefix="Epoch: [{}]".format(epoch))
    end=time.time()
    if args.net_module_flag==-1:
        for i,(data,label) in enumerate(train_loader):
            
            data=data.unsqueeze(dim=1)
            label=label.squeeze()
            data = data.type(torch.FloatTensor)
            label = label.type(torch.LongTensor)
            data=data.cuda(local_rank,non_blocking=True)
            label=label.cuda(local_rank,non_blocking=True)
            
            
            
            output=model(data)
            
            loss = criterion(output, label)
                
            torch.distributed.barrier()
            
            total_output=gather_tensor(output)
            total_label=gather_tensor(label)
            
            result=accuracy(total_output,total_label)
            Accuracy.update(result.Accuracy)
            reduced_loss = reduced_mean(loss, args.nprocs)
            
            losses.update(reduced_loss.item(), data.size(0))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_time.update(time.time() - end)
            end = time.time()
    
            if i % args.print_freq == 0:
    
                progress.display(i)
    else:  
#        size1=args.data1.shape
#        size2=args.data2.shape
#        size3=args.data3.shape
        for idx in range(args.n_data):
            exec('size'+str(idx)+'=args.data'+str(idx)+'.shape')
        for i,(data,label) in enumerate(train_loader):
             n_sample=data.shape[0]
             begin_position=0
             end_position=size0[1]*size0[2]*size0[3]
             model_command_line='output=model('
             for idx in range(args.n_data):
                 exec('data_channel'+str(idx)+'=data[:,begin_position:end_position]')
                 begin_position=end_position
                 if (idx+1) <args.n_data :
                     exec('end_position=end_position+size'+str(idx+1)+'[1]*size'+str(idx+1)+'[2]*size'+str(idx+1)+'[3]')
                 exec('data_channel'+str(idx)+'=data_channel'+str(idx)+'.reshape(n_sample,size'+str(idx)+'[1],size'+str(idx)+'[2],size'+str(idx)+'[3])')
                 exec('data_channel'+str(idx)+'=data_channel'+str(idx)+'.unsqueeze(idx=1).type(torch.FloatTensor).cuda(local_rank,non_blocking=True)')
                 model_command_line=model_command_line+'data_channel'+str(idx)+','
#             data_channel1=data[:,0:size1[1]*size1[2]*size1[3]]
#             data_channel2=data[:,size1[1]*size1[2]*size1[3]:size2[1]*size2[2]*size2[3]+size1[1]*size1[2]*size1[3]]
#             data_channel3=data[:,size2[1]*size2[2]*size2[3]+size1[1]*size1[2]*size1[3]:size2[1]*size2[2]*size2[3]+size1[1]*size1[2]*size1[3]+size3[1]*size3[2]*size3[3]]
#             
#             
#             data_channel1=data_channel1.reshape( n_sample,size1[1],size1[2],size1[3])
#             data_channel2=data_channel2.reshape( n_sample,size2[1],size2[2],size2[3])
#             data_channel3=data_channel3.reshape( n_sample,size3[1],size3[2],size3[3])
             
#             data_channel1=data_channel1.unsqueeze(dim=1)
#             data_channel1 = data_channel1.type(torch.FloatTensor)
#             data_channel1=data_channel1.cuda(local_rank,non_blocking=True)
#             
#             data_channel2=data_channel2.unsqueeze(dim=1)
#             data_channel2 = data_channel2.type(torch.FloatTensor)
#             data_channel2=data_channel2.cuda(local_rank,non_blocking=True)
#
#             data_channel3=data_channel3.unsqueeze(dim=1)
#             data_channel3 = data_channel3.type(torch.FloatTensor)
#             data_channel3=data_channel3.cuda(local_rank,non_blocking=True)
             
             label=label.squeeze()
             label = label.type(torch.LongTensor)
             label=label.cuda(local_rank,non_blocking=True)
             
#             output=model(data_channel1,data_channel2,data_channel3)
             model_command_line=model_command_line[:-1]+')'
             exec(model_command_line)
             loss = criterion(output, label)
                
             torch.distributed.barrier()
                
             total_output=gather_tensor(output)
             total_label=gather_tensor(label)
                
             result=accuracy(total_output,total_label)
             Accuracy.update(result.Accuracy)
             reduced_loss = reduced_mean(loss, args.nprocs)
                
             losses.update(reduced_loss.item(), data.size(0))
                
             optimizer.zero_grad()
             loss.backward()
             optimizer.step()
                
             batch_time.update(time.time() - end)
             end = time.time()
        
             if i % args.print_freq == 0:
        
                progress.display(i)
             
def validate(test_loader, model, criterion, local_rank, args):
    
    model.eval()
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    Accuracy= AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(test_loader), [batch_time, losses, Accuracy ], prefix='Test: ')
    with torch.no_grad():
        end = time.time()
        if args.net_module_flag==-1:
            for i, (data, label) in enumerate(test_loader):
                data=data.unsqueeze(dim=1)
                label=label.squeeze()
                data = data.type(torch.FloatTensor)
                label = label.type(torch.LongTensor)
                data=data.cuda(local_rank,non_blocking=True)
                label=label.cuda(local_rank,non_blocking=True)
    
                # compute output
                output = model(data)
                loss = criterion(output, label)
    
    
    
                torch.distributed.barrier()
                
                total_output=gather_tensor(output)
                total_label=gather_tensor(label)
    
                result=accuracy(total_output,total_label)
                reduced_loss = reduced_mean(loss, args.nprocs)
            
                losses.update(reduced_loss.item(), data.size(0))
                
                batch_time.update(time.time() - end)
                
                Accuracy.update(result.Accuracy)
                end = time.time()
    
                if i % args.print_freq == 0:
                    progress.display(i)
    
    
            return result
        else:
#            size1=args.data1.shape
#            size2=args.data2.shape
#            size3=args.data3.shape
            for idx in range(args.n_data):
                exec('size'+str(idx)+'=args.data'+str(idx)+'.shape')
            for i,(data,label) in enumerate(test_loader):
             n_sample=data.shape[0]
             begin_position=0
             end_position=size0[1]*size0[2]*size0[3]
             model_command_line='output=model('
             for idx in range(args.n_data):
                 exec('data_channel'+str(idx)+'=data[:,begin_position:end_position]')
                 begin_position=end_position
                 if (idx+1) <args.n_data :
                     exec('end_position=end_position+size'+str(idx+1)+'[1]*size'+str(idx+1)+'[2]*size'+str(idx+1)+'[3]')
                 exec('data_channel'+str(idx)+'=data_channel'+str(idx)+'.reshape(n_sample,size'+str(idx)+'[1],size'+str(idx)+'[2],size'+str(idx)+'[3])')
                 exec('data_channel'+str(idx)+'=data_channel'+str(idx)+'.unsqueeze(idx=1).type(torch.FloatTensor).cuda(local_rank,non_blocking=True)')
                 model_command_line=model_command_line+'data_channel'+str(idx)+','
                 
#                 data_channel1=data[:,0:size1[1]*size1[2]*size1[3]]
#                 data_channel2=data[:,size1[1]*size1[2]*size1[3]:size2[1]*size2[2]*size2[3]+size1[1]*size1[2]*size1[3]]
#                 data_channel3=data[:,size2[1]*size2[2]*size2[3]+size1[1]*size1[2]*size1[3]:size2[1]*size2[2]*size2[3]+size1[1]*size1[2]*size1[3]+size3[1]*size3[2]*size3[3]]
#    
#    
#                 n_sample=data.shape[0]
#                    
#                 data_channel1=data_channel1.reshape( n_sample,size1[1],size1[2],size1[3])
#                 data_channel2=data_channel2.reshape( n_sample,size2[1],size2[2],size2[3])
#                 data_channel3=data_channel3.reshape( n_sample,size3[1],size3[2],size3[3])
#                 
#                 data_channel1=data_channel1.unsqueeze(dim=1)
#                 data_channel1 = data_channel1.type(torch.FloatTensor)
#                 data_channel1=data_channel1.cuda(local_rank,non_blocking=True)
#                 
#                 data_channel2=data_channel2.unsqueeze(dim=1)
#                 data_channel2 = data_channel2.type(torch.FloatTensor)
#                 data_channel2=data_channel2.cuda(local_rank,non_blocking=True)
#    
#                 data_channel3=data_channel3.unsqueeze(dim=1)
#                 data_channel3 = data_channel3.type(torch.FloatTensor)
#                 data_channel3=data_channel3.cuda(local_rank,non_blocking=True)
#                     
                 label=label.squeeze()
                 label = label.type(torch.LongTensor)
                 label=label.cuda(local_rank,non_blocking=True)
                    
#                 output=model(data_channel1,data_channel2,data_channel3)
                 model_command_line=model_command_line[:-1]+')'
                 exec(model_command_line)             
                 loss = criterion(output, label)
                
                 torch.distributed.barrier()
                
                 total_output=gather_tensor(output)
                 total_label=gather_tensor(label)
    
                 result=accuracy(total_output,total_label)
                 reduced_loss = reduced_mean(loss, args.nprocs)
            
                 losses.update(reduced_loss.item(), data.size(0))
                
                 batch_time.update(time.time() - end)
                
                 Accuracy.update(result.Accuracy)
                 end = time.time()
    
                 if i % args.print_freq == 0:
                    progress.display(i)
    
    
            return result
        
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
        
        
        
        
        
        
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

class Result(object):
    def __init__(self, Accuracy,TPR,FPR,Precision,Kappa,F1score):
        self.Accuracy=Accuracy
        self.TPR=TPR
        self.FPR=FPR
        self.Precision=Precision
        self.Kappa=Kappa
        self.F1score=F1score
        
def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1**(epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def accuracy(output,label):
    with torch.no_grad():
        
        TP=0
        TN=0
        FP=0
        FN=0
        n_sample=output.shape[0]

        for i in range(n_sample):

            if (output[i,0]<output[i,1]) and (label[i]==1):
                TP=TP+1
            if (output[i,0]>=output[i,1]) and (label[i]==0):
                TN=TN+1
            if (output[i,0]<output[i,1]) and (label[i]==0):  
                FP=FP+1
            if (output[i,0]>=output[i,1]) and (label[i]==1):  
                FN=FN+1
                
        Accuracy = -1
        TPR=-1
        FPR=-1
        Precision=-1
        Recall=-1  
        F1score=-1
        Pe=-1
        Kappa=-1
        
        if TP+TN+FP+FN !=0:
            Accuracy=(TP+TN) / (TP+TN+FP+FN) 
        if TP+FN !=0:
            TPR=TP/(TP+FN) 
        if FP+FN != 0:
            FPR=FP/(FP+TN)  
        if TP+FP != 0:
            Precision=TP/(TP+FP)
        if TP+FN != 0:
            Recall=TP/(TP+FN)
        if Precision+Recall != 0:
            F1score=(2*Precision*Recall)/(Precision+Recall)  
        if TP+TN+FP+FN != 0:
            m=(TP+FP)*(TP+FN)
            n=(TN+FN)*(TN+FP)
            Pe=(n+m)/((TP+TN+FP+FN)**2)
        if 1-Pe !=0:
            Kappa=(Accuracy-Pe)/(1-Pe)

        result=Result(Accuracy=Accuracy,TPR=TPR,FPR=FPR,Precision=Precision, Kappa=Kappa,F1score=F1score)
        
        return result
if __name__ == '__main__':
    main()