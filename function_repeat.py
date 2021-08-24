# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 20:41:11 2021

@author: Ding
"""

import torch

a=torch.tensor([[1,2,3,4],[4,5,6,7],[7,8,9,10]])
a.shape
b=a.repeat(1,1,3)
b.shape
b[0,:,:]

for i in range(3):
    temp_feature=a[i,:]
    c=temp_feature.repeat(1,3)
    c=c.view(3,4)
    print(c)
size=5
def feature_expand(feature,size):
    n_sample=feature.shape[0]
    feature_size=feature.shape[1]
    expand_feature=[]
    for i in range(n_sample):
        current_feature=feature[i,:]
        temp_feature=current_feature.repeat(size,1)
        
        temp_feature=torch.transpose(temp_feature,0,1)
        
        expand_feature.append(temp_feature)
    expand_feature=torch.cat(expand_feature, dim=0)
    output=expand_feature.view(n_sample, feature_size,size)
    return output



c=feature_expand(a,size)
c.shape
c[0,:,:]
d=torch.transpose(c,1,2)
d.shape
d[0,:,:]