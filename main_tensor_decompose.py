# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 10:13:00 2021

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
parser.add_argument('--epochs', default=150, type=int, 
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
    
def kronecker_product(x1,x2):
    n_sample, size1= x1.shape
    n_sample, size2 =x2.shape
    kron_prod=[]
    for i in range (n_sample):
        current_feature1=x1[i,:]
        current_feature2=x2[i,:]
        current_feature1=current_feature1.reshape(1, size1)
        current_feature2=current_feature2.reshape(size2, 1)
        kron_prod.append((current_feature1*current_feature2).view(-1))
    kron_prod=torch.cat( kron_prod, dim=0)
    output=kron_prod.view(n_sample,size1*size2)
    return output
    


def feature_kron(feature1,feature2):
    assert feature1.shape[0] == feature2.shape[0]
    feature_kron=kronecker_product(feature1,feature2)
    return feature_kron
    

    
    
    
class Main_Net(nn.Module):
    def __init__(self,branch1,branch2,compressed1, compressed2):
        super(Main_Net,self).__init__()
        
        self.branch1=branch1
        self.branch2=branch2
        
        self.compressed1=compressed1
        self.compressed2=compressed2
        
        self.fcn1=nn.Sequential(nn.Linear(in_features=64*64, out_features=64, bias=True)
                                )
        
        self.fcn2=nn.Sequential(nn.Linear(in_features=64, out_features=2, bias=True))
    def forward(self,x1,x2):
        
        feature1=self.branch1(x1)
        feature2=self.branch2(x2)
#        n_sample=feature1.shape[0]
#        feature1=feature1.view(n_sample,-1)
#        feature2=feature2.view(n_sample,-1)
        feature1=self.compressed1(feature1)
        feature2=self.compressed2(feature2)
        
        feature=feature_kron(feature1,feature2)
        
        output=self.fcn1(feature)
        output=self.fcn2(output)
        
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


def main():
    args = parser.parse_args()
    
#    args.nprocs = 2
    args.nprocs = torch.cuda.device_count()
#    os.environ['CUDA_VISIBLE_DEVICES']='4,5'
    os.environ['MASTER_ADDR'] ='124.16.75.175'
    os.environ['MASTER_PORT'] = '8888'
    
    data_path1='/home/ding/exp_2/data/Hippocampus_LAD_NC.npy'
    data_path2='/home/ding/exp_2/data/Hippocampus_R_data.npy'
    
    data1=np.load(data_path1)
    data2=np.load(data_path2)


    
    label=np.concatenate((np.ones([200,1],dtype=float),np.zeros([235,1],dtype=float)),0)
    
    args.label=label
    
    args.data1=data1
    args.data2=data2
    
    model1=resnet3d.ResNet(resnet3d.BasicBlock, [1, 1, 1, 1],resnet3d.get_inplanes(),
                     n_input_channels=1,
                     conv1_t_size=27,
                     conv1_t_size2=27,
                     conv1_t_size3=20,
                     conv1_t_stride=1,
                     no_max_pool=False,
                     shortcut_type='B',
                     n_classes=2)
    
    model2=resnet3d.ResNet(resnet3d.BasicBlock, [1, 1, 1, 1],resnet3d.get_inplanes(),
                     n_input_channels=1,
                     conv1_t_size=27,
                     conv1_t_size2=28,
                     conv1_t_size3=21,
                     conv1_t_stride=1,
                     no_max_pool=False,
                     shortcut_type='B',
                     n_classes=2)
    
    args.save_path1='/home/ding/exp_2/result/result_Hippocampus_L_kronecker_compressed_gradient_'
    args.save_path2='/home/ding/exp_2/result/result_Hippocampus_R_kronecker_compressed_gradient_'
    
    args.orginal_params_save_path1='/home/ding/exp_2/data/Hippocampus_L_original_compressed_gradient_'
    args.orginal_params_save_path2='/home/ding/exp_2/data/Hippocampus_R_original_compressed_gradient_'
    
    args.params_save_path1='/home/ding/exp_2/data/Hippocampus_L_kronecker_compressed_gradient_'
    args.params_save_path2='/home/ding/exp_2/data/Hippocampus_R_kronecker_compressed_gradient_'
    
    args.model1=model1
    args.model2=model2
    
    args.save_path_main='/home/ding/exp_2/result/result_kronecker_product_exp_compressed_gradient_'
    
    args.net_module_flag=-1
    
    mp.spawn(main_worker, nprocs=args.nprocs, args=(args.nprocs, args))
    
def main_worker(local_rank,nprocs, args):
    args.local_rank = local_rank
    torch.distributed.init_process_group(backend='nccl', world_size=args.nprocs, rank=local_rank)
    torch.cuda.set_device(local_rank)
    cudnn.benchmark = True
    args.batch_size = int(args.batch_size / args.nprocs)        
    
    kf = StratifiedKFold(n_splits=5,shuffle=True,random_state=2021)
    original_params1=args.model1.state_dict()
    original_params2=args.model2.state_dict()
    
    n_split=0
    result_mat1=np.zeros([5,7])
    result_mat2=np.zeros([5,7])
    result_mat_main=np.zeros([5,7])
    for train_idx, test_idx in kf.split(args.data1,args.label):
        
        args.net_module_flag=-1
        
        current_model1=args.model1
        current_model1.load_state_dict(original_params1)
        current_model1.cuda(local_rank)
        model_para = torch.nn.parallel.DistributedDataParallel(current_model1, device_ids=[local_rank])
        best_acc1= .0
        criterion = nn.CrossEntropyLoss().cuda(local_rank)
        optimizer = torch.optim.SGD(model_para.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        train_loader,test_loader,train_sampler,test_sampler= dataloader_preprocessing(train_idx,test_idx,args,args.data1)     

        for epoch in range(args.epochs):
            
            train_sampler.set_epoch(epoch)
            test_sampler.set_epoch(epoch)
            
            adjust_learning_rate(optimizer, epoch, args)
            
            train(train_loader, model_para, criterion, optimizer, epoch, local_rank, args)
            
            acc1 = validate(test_loader, model_para, criterion, local_rank, args)
            
            is_best = acc1.Accuracy > best_acc1
            best_acc1 = max(acc1.Accuracy, best_acc1)

            if args.local_rank == 0:
                save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model_para.module.state_dict(),
                        'best_acc1': best_acc1,
                    }, is_best)
        
        result_path=args.save_path1+str(n_split)+'time.npy'
        result_mat1[n_split,0]=acc1.Accuracy
        result_mat1[n_split,1]=acc1.TPR
        result_mat1[n_split,2]=acc1.FPR
        result_mat1[n_split,3]=acc1.Precision
        result_mat1[n_split,4]=acc1.F1score
        result_mat1[n_split,5]=acc1.Kappa
        
        np.save(result_path,result_mat1)
#        params_save_path=args.params_save_path1+str(n_split)+'exp_params.pth'
#        torch.save(current_model.state_dict(), params_save_path)
        
        
        
        
        
        current_model2=args.model2
        current_model2.load_state_dict(original_params2)
        current_model2.cuda(local_rank)
        model_para = torch.nn.parallel.DistributedDataParallel(current_model2, device_ids=[local_rank])
        best_acc2= .0
        criterion = nn.CrossEntropyLoss().cuda(local_rank)
        optimizer = torch.optim.SGD(model_para.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        train_loader,test_loader,train_sampler,test_sampler= dataloader_preprocessing(train_idx,test_idx,args,args.data2)     

        for epoch in range(args.epochs):
            
            train_sampler.set_epoch(epoch)
            test_sampler.set_epoch(epoch)
            
            adjust_learning_rate(optimizer, epoch, args)
            
            train(train_loader, model_para, criterion, optimizer, epoch, local_rank, args)
            
            acc2 = validate(test_loader, model_para, criterion, local_rank, args)
            
            is_best = acc2.Accuracy > best_acc2
            best_acc2 = max(acc2.Accuracy, best_acc2)

            if args.local_rank == 0:
                save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model_para.module.state_dict(),
                        'best_acc1': best_acc2,
                    }, is_best)
        
        result_path=args.save_path2+str(n_split)+'time.npy'
        result_mat2[n_split,0]=acc2.Accuracy
        result_mat2[n_split,1]=acc2.TPR
        result_mat2[n_split,2]=acc2.FPR
        result_mat2[n_split,3]=acc2.Precision
        result_mat2[n_split,4]=acc2.F1score
        result_mat2[n_split,5]=acc2.Kappa
        
        np.save(result_path,result_mat2)



        set_gradient(current_model1)
        model_channel1= nn.Sequential(*list(current_model1.children())[:-1])
        

        set_gradient(current_model2)
        model_channel2= nn.Sequential(*list(current_model2.children())[:-1])
        
        
        
        
        criterion = nn.MSELoss().cuda(local_rank)
        compressed_channel1=nn.Sequential(
                                    Compressed_Net(512,64),
                                    Uncompressed_Net(64,512))
        compressed_channel1.cuda(local_rank)
        model_channel1.cuda(local_rank)
        model_para = torch.nn.parallel.DistributedDataParallel(compressed_channel1, device_ids=[local_rank])
        model_para1 = torch.nn.parallel.DistributedDataParallel(model_channel1, device_ids=[local_rank])
        train_loader,test_loader,train_sampler,test_sampler= dataloader_preprocessing(train_idx,test_idx,args,args.data1)     
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model_para.parameters()), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        for epoch in range(args.epochs):
            
            train_sampler.set_epoch(epoch)
            test_sampler.set_epoch(epoch)
            
            adjust_learning_rate(optimizer, epoch, args)
            
            train_compressed(train_loader, model_para, model_para1, criterion, optimizer, epoch, local_rank, args)
        
        
        
        
        criterion = nn.MSELoss().cuda(local_rank)
        compressed_channel2=nn.Sequential(
                                    Compressed_Net(512,64),
                                    Uncompressed_Net(64,512))
        compressed_channel2.cuda(local_rank)
        model_channel2.cuda(local_rank)
        model_para = torch.nn.parallel.DistributedDataParallel(compressed_channel2, device_ids=[local_rank])
        model_para2 = torch.nn.parallel.DistributedDataParallel(model_channel2, device_ids=[local_rank])
        train_loader,test_loader,train_sampler,test_sampler= dataloader_preprocessing(train_idx,test_idx,args,args.data2)     
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model_para.parameters()), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        for epoch in range(args.epochs):
            
            train_sampler.set_epoch(epoch)
            test_sampler.set_epoch(epoch)
            
            adjust_learning_rate(optimizer, epoch, args)
            
            train_compressed(train_loader, model_para, model_para2, criterion, optimizer, epoch, local_rank, args)
        
        
        
        
        
        compress_net1=nn.Sequential(*list(compressed_channel1.children())[:-1])
        compress_net2=nn.Sequential(*list(compressed_channel2.children())[:-1])
        
        set_gradient(compress_net1, n_layers=0)
        set_gradient(compress_net2, n_layers=0)
        current_model=Main_Net(model_channel1,model_channel2,compress_net1,compress_net2)
        
        
        current_model.cuda(local_rank)
        model_para = torch.nn.parallel.DistributedDataParallel(current_model, device_ids=[local_rank])
        best_acc3= .0
        criterion = nn.CrossEntropyLoss().cuda(local_rank)
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model_para.parameters()), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        flatten_data=data_synthesize(args.data1,args.data2)
        train_loader,test_loader,train_sampler,test_sampler= dataloader_preprocessing(train_idx,test_idx,args,flatten_data)     
        args.net_module_flag=1
        for epoch in range(args.epochs):
            
            train_sampler.set_epoch(epoch)
            test_sampler.set_epoch(epoch)
            
            adjust_learning_rate(optimizer, epoch, args)
            
            train(train_loader, model_para, criterion, optimizer, epoch, local_rank, args)
            
            acc3 = validate(test_loader, model_para, criterion, local_rank, args)
            
            is_best = acc3.Accuracy > best_acc3
            best_acc3= max(acc3.Accuracy, best_acc3)

            if args.local_rank == 0:
                save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model_para.module.state_dict(),
                        'best_acc1': best_acc3,
                    }, is_best)
        
        result_path=args.save_path_main+str(n_split)+'time.npy'
        result_mat_main[n_split,0]=acc3.Accuracy
        result_mat_main[n_split,1]=acc3.TPR
        result_mat_main[n_split,2]=acc3.FPR
        result_mat_main[n_split,3]=acc3.Precision
        result_mat_main[n_split,4]=acc3.F1score
        result_mat_main[n_split,5]=acc3.Kappa
        np.save(result_path,result_mat_main)
        
        n_split=n_split+1
        
        
def set_gradient(model,n_layers=38):
        params = list(model.named_parameters())
        layer_list=[]
        for i in range(n_layers):
            layer_list.append(params[i][0])
        
        for k,v in model.named_parameters():
            if k not in layer_list:
                v.requires_grad=False
                
def data_synthesize(data1,data2):
    n_sample=data1.shape[0]
    x1=data1.shape[1]
    y1=data1.shape[2]
    z1=data1.shape[3]
    
    data1_flatten=data1.reshape(n_sample,x1*y1*z1)
    x2=data2.shape[1]
    y2=data2.shape[2]
    z2=data2.shape[3]
    data2_flatten=data2.reshape(n_sample,x2*y2*z2)
    data=np.concatenate((data1_flatten,data2_flatten),axis=1)
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
        size1=args.data1.shape
        size2=args.data2.shape
        for i,(data,label) in enumerate(train_loader):
             data_channel1=data[:,:size1[1]*size1[2]*size1[3]]
             data_channel2=data[:,size1[1]*size1[2]*size1[3]:]
             
             n_sample=data.shape[0]
             
             data_channel1=data_channel1.reshape( n_sample,size1[1],size1[2],size1[3])
             data_channel2=data_channel2.reshape( n_sample,size2[1],size2[2],size2[3])
             
             data_channel1=data_channel1.unsqueeze(dim=1)
             data_channel1 = data_channel1.type(torch.FloatTensor)
             data_channel1=data_channel1.cuda(local_rank,non_blocking=True)
             
             data_channel2=data_channel2.unsqueeze(dim=1)
             data_channel2 = data_channel2.type(torch.FloatTensor)
             data_channel2=data_channel2.cuda(local_rank,non_blocking=True)
             
             label=label.squeeze()
             label = label.type(torch.LongTensor)
             label=label.cuda(local_rank,non_blocking=True)
             
             output=model(data_channel1,data_channel2)
             
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
            size1=args.data1.shape
            size2=args.data2.shape
            for i,(data,label) in enumerate(test_loader):
                data_channel1=data[:,:size1[1]*size1[2]*size1[3]]
                data_channel2=data[:,size1[1]*size1[2]*size1[3]:]
             
                n_sample=data.shape[0]
             
                data_channel1=data_channel1.reshape( n_sample,size1[1],size1[2],size1[3])
                data_channel2=data_channel2.reshape( n_sample,size2[1],size2[2],size2[3])
                 
                data_channel1=data_channel1.unsqueeze(dim=1)
                data_channel1 = data_channel1.type(torch.FloatTensor)
                data_channel1=data_channel1.cuda(local_rank,non_blocking=True)
                 
                data_channel2=data_channel2.unsqueeze(dim=1)
                data_channel2 = data_channel2.type(torch.FloatTensor)
                data_channel2=data_channel2.cuda(local_rank,non_blocking=True)
                 
                label=label.squeeze()
                label = label.type(torch.LongTensor)
                label=label.cuda(local_rank,non_blocking=True)
                
                output=model(data_channel1,data_channel2)
             
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
            if (output[i,0]>output[i,1]) and (label[i]==0):
                TN=TN+1
            if (output[i,0]<output[i,1]) and (label[i]==0):  
                FP=FP+1
            if (output[i,0]>output[i,1]) and (label[i]==1):  
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