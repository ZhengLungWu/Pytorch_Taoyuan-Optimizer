#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 11:23:02 2019
AdaGrad
@author: wuzhenglung
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib
import matplotlib.animation as animation
import SGD


def run_AdaGrad(eta,x_ini,y_ini,epoch):
    '''
    lr=learning rate
        
    '''
    eps=1e-8
    xt=x_ini
    yt=y_ini
    #plotFx(xt,yt)
    xs=[]
    ys=[]
    xs.append(xt)
    ys.append(yt)
    Gxti=eps
    Gyti=eps
     
    for i in range(epoch):
        dx,dy=dF(xt,yt)
        Gxti+=dx**2
        Gyti+=dy**2
        xt=xt-(eta/(Gxti**0.5))*dx
        yt=yt-(eta/(Gyti**0.5))*dy
        xs.append(xt)
        ys.append(yt)
    xs=np.asarray(xs)
    ys=np.asarray(ys)
    #print(xs)
    plotPath(xs,ys,"AdaGrad")

if __name__=="__main__":            
    run_AdaGrad(0.1,-.5,-4.4,1000)