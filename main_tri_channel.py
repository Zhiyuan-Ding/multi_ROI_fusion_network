# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 19:55:18 2021

@author: Ding
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 20:55:44 2021

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
parser.add_argument('--epochs', default=3, type=int, 
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

class Main_Net(nn.Module):
    def __init__(self,branch1,branch2, branch3):
        super(Main_Net,self).__init__()
        
        self.branch1=branch1
        self.branch2=branch2
        self.branch3=branch3
        
        self.fcn=nn.Sequential(nn.Linear(in_features=512*3, out_features=2, bias=True))
        
    def forward(self,x1,x2,x3):
        
        feature1=self.branch1(x1)
        feature2=self.branch2(x2)
        feature3=self.branch3(x3)
        n_sample=feature1.shape[0]
        feature1=feature1.view(n_sample,-1)
        feature2=feature2.view(n_sample,-1)
        feature3=feature3.view(n_sample,-1)


        features=torch.cat((feature1,feature2,feature3),dim=1)
        
        output=self.fcn(features)
        
        return output
        


def main():
    os.environ['CUDA_VISIBLE_DEVICES']='0,1,2'
    
    args = parser.parse_args()
    
    args.nprocs = torch.cuda.device_count()

    os.environ['MASTER_ADDR'] ='124.16.75.175'
    os.environ['MASTER_PORT'] = '8888'
    
    data_path1='/home/ding/exp_2/data/Hippocampus_LAD_NC.npy'
    data_path2='/home/ding/exp_2/data/Hippocampus_R_data.npy'
    data_path3='/home/ding/exp_2/data/ParaHippocampus_L_data.npy'
    data1=np.load(data_path1)
    data2=np.load(data_path2)
    data3=np.load(data_path3)

    
    label=np.concatenate((np.ones([200,1],dtype=float),np.zeros([235,1],dtype=float)),0)
    
    args.label=label
    
    args.data1=data1
    args.data2=data2
    args.data3=data3
    
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
    
    model3=resnet3d.ResNet(resnet3d.BasicBlock, [1, 1, 1, 1],resnet3d.get_inplanes(),
                 n_input_channels=1,
                 conv1_t_size=23,
                 conv1_t_size2=38,
                 conv1_t_size3=18,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 n_classes=2)    

    
    args.model1=model1
    args.model2=model2
    args.model3=model3
    
    model_channel1= nn.Sequential(*list(model1.children())[:-1])
    model_channel2= nn.Sequential(*list(model2.children())[:-1])
    model_channel3= nn.Sequential(*list(model3.children())[:-1])
    main_model=Main_Net(model_channel1,model_channel2,model_channel3)
    
    args.save_path_main='/home/ding/exp_2/result/result_tri_branch_original_'
    
    args.net_module_flag=1
    
    args.main_model=main_model
    
    mp.spawn(main_worker, nprocs=args.nprocs, args=(args.nprocs, args))
    
def main_worker(local_rank,nprocs, args):
    args.local_rank = local_rank
    torch.distributed.init_process_group(backend='nccl', world_size=args.nprocs, rank=local_rank)
    torch.cuda.set_device(local_rank)
    cudnn.benchmark = True
    args.batch_size = int(args.batch_size / args.nprocs)        
    
    kf = StratifiedKFold(n_splits=5,shuffle=True,random_state=2021)
    
    n_split=0
    result_mat=np.zeros([5,7])
    original_params=args.main_model.state_dict()
    for train_idx, test_idx in kf.split(args.data1,args.label):
        current_model=args.main_model
        current_model.load_state_dict(original_params)
        current_model.cuda(local_rank)
        model_para = torch.nn.parallel.DistributedDataParallel(current_model, device_ids=[local_rank])
        best_acc1= .0
        criterion = nn.CrossEntropyLoss().cuda(local_rank)
        optimizer = torch.optim.SGD(model_para.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        train_loader,test_loader,train_sampler,test_sampler= dataloader_preprocessing_tri_channel(train_idx,test_idx,args)     

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
        
        result_path=args.save_path_main+str(n_split)+'_time.npy'
        result_mat[n_split,0]=acc1.Accuracy
        result_mat[n_split,1]=acc1.TPR
        result_mat[n_split,2]=acc1.FPR
        result_mat[n_split,3]=acc1.Precision
        result_mat[n_split,4]=acc1.F1score
        result_mat[n_split,5]=acc1.Kappa
        
        np.save(result_path,result_mat)
        
        n_split=n_split+1
        
        
def set_gradient(model,n_layers=38):
        params = list(model.named_parameters())
        layer_list=[]
        for i in range(n_layers):
            layer_list.append(params[i][0])
        
        for k,v in model.named_parameters():
            if k not in layer_list:
                v.requires_grad=False
                
def data_synthesize(data1,data2,data3):
    n_sample=data1.shape[0]
    x1=data1.shape[1]
    y1=data1.shape[2]
    z1=data1.shape[3]
    
    data1_flatten=data1.reshape(n_sample,x1*y1*z1)
    x2=data2.shape[1]
    y2=data2.shape[2]
    z2=data2.shape[3]
    data2_flatten=data2.reshape(n_sample,x2*y2*z2)
    
    
    x3=data3.shape[1]
    y3=data3.shape[2]
    z3=data3.shape[3]
    data3_flatten=data3.reshape(n_sample,x3*y3*z3)
    data=np.concatenate((data1_flatten,data2_flatten,data3_flatten),axis=1)
    return data
    
    
def dataloader_preprocessing_tri_channel(train_idx,test_idx,args):
        total_data=data_synthesize(args.data1,args.data2,args.data3)
        
        train_data=total_data[train_idx]
        train_label=args.label[train_idx]
        train_data=np.array(train_data)
        train_label=np.array(train_label)
        train_data=torch.from_numpy(train_data)
        train_label=torch.from_numpy(train_label)
        
        
        test_data=total_data[test_idx]
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
        size3=args.data3.shape
        for i,(data,label) in enumerate(train_loader):

             data_channel1=data[:,0:size1[1]*size1[2]*size1[3]]
             data_channel2=data[:,size1[1]*size1[2]*size1[3]:size2[1]*size2[2]*size2[3]+size1[1]*size1[2]*size1[3]]
             data_channel3=data[:,size2[1]*size2[2]*size2[3]+size1[1]*size1[2]*size1[3]:size2[1]*size2[2]*size2[3]+size1[1]*size1[2]*size1[3]+size3[1]*size3[2]*size3[3]]
             n_sample=data.shape[0]
#             print(data_channel1.shape)
#             print(data_channel2.shape)
#             print(data_channel3.shape)
             data_channel1=data_channel1.reshape( n_sample,size1[1],size1[2],size1[3])
             data_channel2=data_channel2.reshape( n_sample,size2[1],size2[2],size2[3])
             data_channel3=data_channel3.reshape( n_sample,size3[1],size3[2],size3[3])
             
             data_channel1=data_channel1.unsqueeze(dim=1)
             data_channel1 = data_channel1.type(torch.FloatTensor)
             data_channel1=data_channel1.cuda(local_rank,non_blocking=True)
             
             data_channel2=data_channel2.unsqueeze(dim=1)
             data_channel2 = data_channel2.type(torch.FloatTensor)
             data_channel2=data_channel2.cuda(local_rank,non_blocking=True)

             data_channel3=data_channel3.unsqueeze(dim=1)
             data_channel3 = data_channel3.type(torch.FloatTensor)
             data_channel3=data_channel3.cuda(local_rank,non_blocking=True)
             
             label=label.squeeze()
             label = label.type(torch.LongTensor)
             label=label.cuda(local_rank,non_blocking=True)
             
             output=model(data_channel1,data_channel2,data_channel3)
             
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
            size3=args.data3.shape
            for i,(data,label) in enumerate(test_loader):
             data_channel1=data[:,0:size1[1]*size1[2]*size1[3]]
             data_channel2=data[:,size1[1]*size1[2]*size1[3]:size2[1]*size2[2]*size2[3]+size1[1]*size1[2]*size1[3]]
             data_channel3=data[:,size2[1]*size2[2]*size2[3]+size1[1]*size1[2]*size1[3]:size2[1]*size2[2]*size2[3]+size1[1]*size1[2]*size1[3]+size3[1]*size3[2]*size3[3]]


             n_sample=data.shape[0]
                
             data_channel1=data_channel1.reshape( n_sample,size1[1],size1[2],size1[3])
             data_channel2=data_channel2.reshape( n_sample,size2[1],size2[2],size2[3])
             data_channel3=data_channel3.reshape( n_sample,size3[1],size3[2],size3[3])
             
             data_channel1=data_channel1.unsqueeze(dim=1)
             data_channel1 = data_channel1.type(torch.FloatTensor)
             data_channel1=data_channel1.cuda(local_rank,non_blocking=True)
             
             data_channel2=data_channel2.unsqueeze(dim=1)
             data_channel2 = data_channel2.type(torch.FloatTensor)
             data_channel2=data_channel2.cuda(local_rank,non_blocking=True)

             data_channel3=data_channel3.unsqueeze(dim=1)
             data_channel3 = data_channel3.type(torch.FloatTensor)
             data_channel3=data_channel3.cuda(local_rank,non_blocking=True)
                 
             label=label.squeeze()
             label = label.type(torch.LongTensor)
             label=label.cuda(local_rank,non_blocking=True)
                
             output=model(data_channel1,data_channel2,data_channel3)
             
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
            FPR=FP/(FP+FN)  
        if TP+FP != 0:
            Precision=TP/(TP+FP)
        if TP+FN != 0:
            Recall=TP/(TP+FN)
        if Precision+Recall != 0:
            F1score=(2*Precision*Recall)/(Precision+Recall)  
        if TP+TN+FP+FN != 0:
            Pe=(TN+TP)/((TP+TN+FP+FN)**2)
        if 1-Pe !=0:
            Kappa=(Accuracy-Pe)/(1-Pe)

        result=Result(Accuracy=Accuracy,TPR=TPR,FPR=FPR,Precision=Precision, Kappa=Kappa,F1score=F1score)
        
        return result
if __name__ == '__main__':
    main()