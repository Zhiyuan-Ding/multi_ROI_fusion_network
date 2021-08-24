# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 15:47:54 2021

@author: Ding
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 22:50:13 2021

@author: Ding
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 17:08:09 2021

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
import CU_Net3d
import torch.backends.cudnn as cudnn
from sklearn.model_selection import StratifiedKFold
from torch.utils.data.dataset import Dataset
import torch.multiprocessing as mp
#settings
parser = argparse.ArgumentParser(description='single brain region model')

parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--epochs', default=300, type=int, 
                    help='number of total epochs to run')
parser.add_argument('--batch_size',default=100, type=int,
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
#parser.add_argument('data', default=None, type=float, 
#                    help='dataset, including train and test set and their labels, which will be segmented by K fold. ')



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
    def __init__(self, feature_extraction, classifier):
        super(Main_Net, self).__init__()
        self.feature_extraction=feature_extraction
        self.classifier=classifier
        
    def forward(self, x, y):
#        print(x.shape)
#        print(y.shape)
        out=self.feature_extraction(x,y)
#        print(out.shape)
        out=self.classifier(out)
#        print(out.shape)
        return out
   
    
    
def main():
    args = parser.parse_args()
    args.nprocs = torch.cuda.device_count()
    args.nprocs = 5
    os.environ['CUDA_VISIBLE_DEVICES']='1,2,3,4,5'
    os.environ['MASTER_ADDR'] ='124.16.75.175'
    os.environ['MASTER_PORT'] = '12345'
    data_path1='/home/ding/exp_2/data/Hippocampus_LAD_NC.npy'
    data_path2='/home/ding/exp_2/data/Hippocampus_R_data.npy'
    
    data1=np.load(data_path1)
    data2=np.load(data_path2)

    args.data1=data1
    args.data2=data2
    
    label=np.concatenate((np.ones([200,1],dtype=float),np.zeros([235,1],dtype=float)),0)
    args.label=label

    classifier=resnet3d.ResNet(resnet3d.BasicBlock, [1, 1, 1, 1],resnet3d.get_inplanes(),
                         n_input_channels=1,
                         conv1_t_size=27,
                         conv1_t_size2=27,
                         conv1_t_size3=20,
                         conv1_t_stride=1,
                         no_max_pool=False,
                         shortcut_type='B',
                         n_classes=2)
    feature_extraction=CU_Net3d.CUNet()
    
    total_model=Main_Net(feature_extraction, classifier)
    mp.spawn(main_worker, nprocs=args.nprocs, args=(args.nprocs, args, total_model))
    

def main_worker(local_rank,nprocs, args, model):
    args.local_rank = local_rank
    torch.distributed.init_process_group(backend='nccl', world_size=args.nprocs, rank=local_rank)
    torch.cuda.set_device(local_rank)
    cudnn.benchmark = True
    args.batch_size = int(args.batch_size / args.nprocs)        
    
    kf = StratifiedKFold(n_splits=5,shuffle=True,random_state=2021)
    original_params=model.state_dict()
    n_split=0
    result_mat=np.zeros([5,7])
    
    for train_idx, test_idx in kf.split(args.data1,args.label):
        
        model.load_state_dict(original_params)
        
        model.cuda(local_rank)
        model_para = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        best_acc1= .0


    
        criterion = nn.CrossEntropyLoss().cuda(local_rank)
        optimizer = torch.optim.SGD(model_para.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        flatten_data=data_synthesize(args.data1,args.data2)
        train_loader,test_loader,train_sampler,test_sampler= dataloader_preprocessing(train_idx,test_idx,args,flatten_data)     

        
        
        if args.evaluate:
            validate(test_loader, model_para, criterion, local_rank, args)
            return
        
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
        
        result_path='/home/ding/exp_2/result/result_Hippocampus_CUNET_'+str(n_split)+'time.npy'
        result_mat[n_split,0]=acc1.Accuracy
        result_mat[n_split,1]=acc1.TPR
        result_mat[n_split,2]=acc1.FPR
        result_mat[n_split,3]=acc1.Precision
        result_mat[n_split,4]=acc1.F1score
        result_mat[n_split,5]=acc1.Kappa
        n_split=n_split+1
        np.save(result_path,result_mat)
        
def train(train_loader, model, criterion, optimizer, epoch, local_rank, args):
    model.train()
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    Accuracy=AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(train_loader), [batch_time, losses, Accuracy],
                             prefix="Epoch: [{}]".format(epoch))
    end=time.time()
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

def validate(test_loader, model, criterion, local_rank, args):
    
    model.eval()
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    Accuracy= AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(test_loader), [batch_time, losses, Accuracy ], prefix='Test: ')
   
    with torch.no_grad():
        end = time.time()
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
            Pe=(TN+TP)/((TP+TN+FP+FN)**2)
        if 1-Pe !=0:
            Kappa=(Accuracy-Pe)/(1-Pe)

        result=Result(Accuracy=Accuracy,TPR=TPR,FPR=FPR,Precision=Precision, Kappa=Kappa,F1score=F1score)
        
        return result
        
if __name__ == '__main__':
    main()