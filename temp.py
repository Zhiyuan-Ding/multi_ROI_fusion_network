# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 15:45:13 2021

@author: Ding
"""

import tensorly
import torch
import numpy as np
a=torch.tensor([[1,2,3],[4,5,6]])
b=torch.tensor([[1,2,3],[4,5,6]])
c=tensorly.kron(a,b)
d=tensorly.kron(a,b)
d[1,:]
kronecker_product(a,b)
mean_result=np.mean(result_resnet10Hippocampus_R4time,axis=0)
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
    print(kron_prod)
    kron_prod=torch.cat( kron_prod, dim=0)
    output=kron_prod.view(n_sample,size1*size2)
    return output
    
    
        