# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 10:39:46 2021

@author: Ding
"""
import torch
import torch.nn as nn
from torchsummary import summary

summary(model1, (1,27,27,20), device="cpu")
summary(model2, (1,27,28,21), device="cpu")
print(model1)
resnet_layer = nn.Sequential(*list(model1.children())[-3:-1],
                             nn.Flatten(),
                             nn.Linear(512,2,bias=True))
summary(resnet_layer, (256,4,1,1), device="cpu")
#resnet_layer[0][0].conv1.in_channels=512
resnet_layer[0][0].conv1=nn.Conv3d(512,512,kernel_size=(3,3,3),stride=(2,2,2), padding=(1,1,1),bias=False)
resnet_layer[0][0].downsample[0]=nn.Conv3d(512, 512, kernel_size=(1, 1, 1), stride=(2, 2, 2), bias=False)
summary(resnet_layer, (512,4,1,1), device="cpu")


layer1=nn.Sequential(*list(model1.children())[-10:-6])
summary(layer1, (1,27,27,20), device="cpu")
layer2=nn.Sequential(*list(model1.children())[-6:-5])
summary(layer2, (64,14,2,2), device="cpu")
layer3=nn.Sequential(*list(model1.children())[-5:-4])
summary(layer3, (64,14,2,2), device="cpu")
layer4=nn.Sequential(*list(model1.children())[-4:-3])
summary(layer4, (128,7,1,1), device="cpu")
layer5=nn.Sequential(*list(model1.children())[-3:-2])
transpose_layer5=nn.Sequential(nn.ConvTranspose3d(64,512,kernel_size=1, stride=1, padding=0))
summary(transpose_layer5, (64,14,2,2), device="cpu")
summary(layer5, (256,4,1,1), device="cpu")
layer6=nn.Sequential(*list(model1.children())[-2:-1],
                                  nn.Flatten(),
                                  nn.Linear(512,2))
summary(layer6, (512,2,1,1), device="cpu")

summary(total_model, (1,27,27,20), device="cpu")

lam_in = nn.Parameter(torch.Tensor([0.01]))


exp_layer=nn.Sequential(nn.Conv3d(1,10,kernel_size=1))

summary(exp_layer, (1,27,27,20), device="cpu")