# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 15:58:12 2021

@author: Ding
"""
import torch.nn as nn
import torch
x=torch.tensor([[1,2],[3,4],[5,6]], dtype= float)
centers=nn.Parameter(torch.randn((2,2),dtype= float))
distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, num_classes) + \
                  torch.pow(centers, 2).sum(dim=1, keepdim=True).expand(num_classes, batch_size).t()
distmat.addmm_(1, -2, x, centers.t())
class CenterLoss(nn.Module):
    def __init__(self, num_classes=2, feat_dim=2, use_gpu=True, local_rank):
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
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss
        
m=1
n=2
       