# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 17:09:50 2021

@author: Ding
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 10:28:40 2021

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
#parser.add_argument('data', default=None, type=float, 
#                    help='dataset, including train and test set and their labels, which will be segmented by K fold. ')


class CenterLoss(nn.Module):
    def __init__(self, num_classes=2, feat_dim=2, use_gpu=True, local_rank=None):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        self.local_rank=local_rank
        
        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda(self.local_rank))
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
            
    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda(self.local_rank)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss
    
    
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



def main():
    args = parser.parse_args()
    args.nprocs = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] ='124.16.75.175'
    os.environ['MASTER_PORT'] = '8888'
    data_path='/home/ding/exp_2/data/Hippocampus_R.npy'
    data=np.load(data_path)
    args.alpha=0.05

    
    label=np.concatenate((np.ones([200,1],dtype=float),np.zeros([235,1],dtype=float)),0)
    args.label=label
    data=InputDataset(data,label)
    args.data=data
    model=resnet3d.ResNet(resnet3d.BasicBlock, [1, 1, 1, 1],resnet3d.get_inplanes(),
                         n_input_channels=1,
                         conv1_t_size=27,
                         conv1_t_size2=28,
                         conv1_t_size3=21,
                         conv1_t_stride=1,
                         no_max_pool=False,
                         shortcut_type='B',
                         n_classes=2)
    mp.spawn(main_worker, nprocs=args.nprocs, args=(args.nprocs, args, model))
    

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
    for train_idx, test_idx in kf.split(args.data.Data,args.label):
        
        model.load_state_dict(original_params)
        
        model.cuda(local_rank)
        model_para = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        best_acc1= .0


    
        criterion_center = CenterLoss(local_rank=local_rank).cuda(local_rank)
        criterion = nn.CrossEntropyLoss().cuda(local_rank)
        optimizer_center = torch.optim.SGD(criterion_center.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        optimizer = torch.optim.SGD(model_para.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        
        train_data=args.data.Data[train_idx]
        train_label=args.data.Label[train_idx]
        train_data=np.array(train_data)
        train_label=np.array(train_label)
        train_data=torch.from_numpy(train_data)
        train_label=torch.from_numpy(train_label)
        
        
        test_data=args.data.Data[test_idx]
        test_label=args.data.Label[test_idx]
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
        
        
#        if args.evaluate:
#            validate(test_loader, model_para, criterion, local_rank, args)
#            return
        
        for epoch in range(args.epochs):
            
            train_sampler.set_epoch(epoch)
            test_sampler.set_epoch(epoch)
            
            adjust_learning_rate(optimizer, epoch, args)
            
            train(train_loader, model_para, criterion, criterion_center, optimizer,optimizer_center, epoch, local_rank, args)
            
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
        
        result_path='/home/ding/exp_2/result/result_center_loss_01_Hippocampus_R_'+str(n_split)+'time.npy'
        result_mat[n_split,0]=acc1.Accuracy
        result_mat[n_split,1]=acc1.TPR
        result_mat[n_split,2]=acc1.FPR
        result_mat[n_split,3]=acc1.Precision
        result_mat[n_split,4]=acc1.F1score
        result_mat[n_split,5]=acc1.Kappa
        n_split=n_split+1
        np.save(result_path,result_mat)
        
def train(train_loader, model, criterion,criterion_center, optimizer,optimizer_center, epoch, local_rank, args):
    model.train()
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    Accuracy=AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(train_loader), [batch_time, losses, Accuracy],
                             prefix="Epoch: [{}]".format(epoch))
    end=time.time()
    for i,(data,label) in enumerate(train_loader):
#        print(i)
        data=data.unsqueeze(dim=1)
        label=label.squeeze()
        data = data.type(torch.FloatTensor)
        label = label.type(torch.LongTensor)
        data=data.cuda(local_rank,non_blocking=True)
        label=label.cuda(local_rank,non_blocking=True)
        
        
        
        output=model(data)
        loss = criterion_center(output, label) * args.alpha + criterion(output, label)
#        loss = criterion(output, label)
            
        torch.distributed.barrier()
        
        total_output=gather_tensor(output)
        total_label=gather_tensor(label)
        
        result=accuracy(total_output,total_label)
        Accuracy.update(result.Accuracy)
        reduced_loss = reduced_mean(loss, args.nprocs)
        
        losses.update(reduced_loss.item(), data.size(0))
        optimizer_center.zero_grad()
        optimizer.zero_grad()
        loss.backward()
        for param in criterion_center.parameters():
            param.grad.data *= (1./args.alpha)
        optimizer.step()
        optimizer_center.step()

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