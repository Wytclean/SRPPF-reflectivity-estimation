from unicodedata import name
from unittest import TestLoader
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
    def __init__(self,droot,lroot,hroot,train=True,partion=0.8):
        super().__init__()
        self.datan=np.array(sorted(glob.glob(os.path.join(droot,'*.dat'))))#将sesmic data存入一个array
        self.labeln=np.array(sorted(glob.glob(os.path.join(lroot,'*.dat'))))#将标签存入一个array
        self.hn=np.array(sorted(glob.glob(os.path.join(hroot,'*.dat'))))
        n=np.arange(len(self.datan))#创建一个跟datan一样长的arange
        np.random.shuffle(n)#为了后期标签和数据能够成对的一样的打乱顺序方便调试
        self.datan=self.datan[n]#按照n的方式进行打乱，下同
        self.labeln=self.labeln[n]
        self.hn=self.hn[n]
        if train:
            self.datan=self.datan[:int(partion*len(n))]
            self.labeln=self.labeln[:int(partion*len(n))]
            self.hn=self.hn[:int(partion*len(n))]                    
        else:
            self.datan=self.datan[int(partion*len(n)):]
            self.labeln=self.labeln[int(partion*len(n)):]
            self.hn=self.hn[int(partion*len(n)):]
            print(self.datan)

    def __getitem__(self, index):
        d,label,h=self.loade(index)
        return d,label,h

    def loade(self,index):
        n1,n2,n3=256,256,256
        m1,m2,m3=128,128,128
        # index=index//3
        dx=np.fromfile(self.datan[index],np.float32).reshape(n3,n2,n1)
        lx=np.fromfile(self.labeln[index],np.float32).reshape(n3,n2,n1)
        hx=np.fromfile(self.hn[index],np.float32).reshape(n3,n2,n1)
        k1 = random.randint(0,127)
        k2 = random.randint(0,127)
        k3 = random.randint(0,127)
        dx = dx[k3:k3+m3,k2:k2+m2,k1:k1+m1]
        lx = lx[k3:k3+m3,k2:k2+m2,k1:k1+m1]
        hx = hx[k3:k3+m3,k2:k2+m2,k1:k1+m1]
        #data normalization
        dm = np.mean(dx)
        lm = np.mean(lx)
        ds = np.std(dx)
        ls = np.std(lx)
        dx = dx-dm
        dx = dx/ds
        lx = lx-lm
        lx = lx/ls
        hx = np.clip(hx,0,1)
        #transpose the matrix from[n3][n2][n1] to [n1][n2][n3]
        # dx = np.transpose(dx,(1,0,2))
        # lx = np.transpose(lx,(1,0,2))
        # hx = np.transpose(hx,(1,0,2))
        a = 3 #number of data argumentations
        X = np.zeros((a,1,m1,m2,m3),dtype=np.single)
        Y = np.zeros((a,1,m1,m2,m3),dtype=np.single)
        Z = np.zeros((a,1,m1,m2,m3),dtype=np.single)
        X[0,0,] = dx
        Y[0,0,] = lx
        Z[0,0,] = hx
        X[1,0,] = np.flipud(dx)
        Y[1,0,] = np.flipud(lx)
        Z[1,0,] = np.flipud(hx)
        #randomly rotate the 3D array around the vertical axis 
        #by 90, 180, or 270 degrees
        i = random.randint(1,3)
        X[2,0,] =np.rot90(dx,i,(1,2))
        Y[2,0,] =np.rot90(lx,i,(1,2))
        Z[2,0,] =np.rot90(hx,i,(1,2))
        # k = random.randint(0,2)
        k = 0
        #print(np.min(X[k]))
        #print(np.min(Y[k]))
        
        return X[k], Y[k], Z[k]
        # return X, Y


    def __len__(self):
        # return len(self.datan)*3
        return len(self.datan)

# if __name__=="__main__":
#     droot='/home/ytwang/code/inversion/testPath/odata/odata/nx'
#     lroot='/home/ytwang/code/inversion/testPath/odata/odata/rx'
#     hroot='/home/ytwang/code/inversion/testPath/odata/odata/hx'
#     test_loade=Seismicloader(droot,lroot,hroot,False)
#     X,Y,Z=test_loade.__getitem__(0)
#     X.tofile('/home/ytwang/code/inversion/testPath/xyz/x.dat')
#     Y.tofile('/home/ytwang/code/inversion/testPath/xyz/y.dat')
#     Z.tofile('/home/ytwang/code/inversion/testPath/xyz/z.dat')    

