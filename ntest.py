import numpy as np
import os
import glob
import torch
import random

from torch.utils.data import Dataset
from utils import *

import matplotlib.pyplot as plt
import sys


# class GetLoader(Dataset):
    
#     def __init__(self,droot,lroot,tranin=True,partion=0.8) -> None:
#         super().__init__()
#         self.data=sorted(glob.glob(os.path.join(droot, '*.dat')))
#         self.label=sorted(glob.glob(os.path.join(lroot, '*.dat')))
#     def __getitem__(self,index):
#         datad=np.fromfile(self.data[index],dtype=np.float32).reshape(1,1,256,256)
#         latad=np.fromfile(self.label[index],dtype=np.float32).reshape(1,1,256,256)
#         return torch.from_numpy(datad),torch.from_numpy(latad)
#     def __len__(self):
#         return len(self.data)
# source_data="/home/ytw/yting/2dnx/"
# source_label="/home/ytw/yting/2drx/"
# da=GetLoader(source_data,source_label)

class Seismicloader(Dataset):
    def __init__(self,droot,lroot,train=True,partion=0.8):
        super().__init__()
        self.datan=np.array(sorted(glob.glob(os.path.join(droot,'*.dat'))))#将sesmic data存入一个array
        self.labeln=np.array(sorted(glob.glob(os.path.join(lroot,'*.dat'))))#将标签存入一个array
        n=np.arange(len(self.datan))#创建一个跟datan一样长的arange
        np.random.shuffle(n)#为了后期标签和数据能够成对的一样的打乱顺序方便调试
        self.datan=self.datan[n]#按照n的方式进行打乱，下同
        self.labeln=self.labeln[n]
        if train:
            self.datan=self.datan[:int(partion*len(n))]
            self.labeln=self.labeln[:int(partion*len(n))]
        else:
            self.datan=self.datan[int(partion*len(n)):]
            self.labeln=self.labeln[int(partion*len(n)):]

    def __getitem__(self, index):
        d,label=self.loade(index)
        d=d[np.newaxis,:,:,:]
        label=label[np.newaxis,:,:,:]
        return d,label

    def loade(self,index):
        n1,n2,n3=256,256,256
        dx=np.fromfile(self.datan[index],np.float32).reshape(n3,n2,n1)
        lx=np.fromfile(self.labeln[index],np.float32).reshape(n3,n2,n1)
        #data normalization
        dm = np.mean(dx)
        lm = np.mean(lx)
        ds = np.std(dx)
        ls = np.std(lx)
        dx = dx-dm
        dx = dx/ds
        lx = lx-lm
        lx = lx/ls
        #transpose the matrix from[n3][n2][n1] to [n1][n2][n3]
        dx = np.transpose(dx)
        lx = np.transpose(lx)
        a = 3 #number of data argumentations
        X = np.zeros((a,n1,n2,n3),dtype=np.single)
        Y = np.zeros((a,n1,n2,n3),dtype=np.single)
        X[0,] = dx
        Y[0,] = lx
        X[1,] = np.flipud(dx)
        Y[1,] = np.flipud(lx)
        #randomly rotate the 3D array around the vertical axis 
        #by 90, 180, or 270 degrees
        i = random.randint(1,3)
        X[2,...] =np.rot90(dx,i,(1,2))
        Y[2,...] =np.rot90(lx,i,(1,2))
        k = random.randint(0,2)
        return X[k], Y[k]


    def __len__(self):
        return(len(self.datan))



