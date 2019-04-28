#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 14:14:51 2019
Adam AdaMax
@author: wuzhenglung
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib
import matplotlib.animation as animation
import SGD
import math


def run_Adam(beta1,beta2,eta,x_ini,y_ini,epoch):
    eps=1e-6
    
    xt=x_ini
    yt=y_ini
    
    
    mx=0
    my=0
    
    vx=0
    vy=0
    
    xs=[]
    ys=[]
    xs.append(xt)
    ys.append(yt)
    
    for i in range(epoch):
        dx,dy=dF(xt,yt)
        mx=beta1*mx+(1-beta1)*dx
        my=beta1*my+(1-beta1)*dy
        
        vx=beta2*vx+(1-beta2)*dx**2
        vy=beta2*vy+(1-beta2)*dy**2
        if i!=0:
            mmx=mx/(1-beta1**i)
            mmy=my/(1-beta1**i)
            
            vvx=vx/(1-beta2**i)
            vvy=vy/(1-beta2**i)
        else:
            mmx=0
            mmy=0
            vvx=0
            vvy=0
        
        xt=xt-eta*mmx/(vvx**0.5+eps)
        yt=yt=eta*mmy/(vvy**0.5+eps)
        xs.append(xt)
        ys.append(yt)
    xs=np.asarray(xs)
    ys=np.asarray(ys)
    #print(xs)
    plotPath(xs,ys,"Adam")
        
def run_AdaMax(beta1,beta2,eta,x_ini,y_ini,epoch):
    eps=1e-6
    
    xt=x_ini
    yt=y_ini
    
    
    mx=0
    my=0
    
    ux=0.
    uy=0
    
    xs=[]
    ys=[]
    xs.append(xt)
    ys.append(yt)
    
    for i in range(epoch):
        dx,dy=dF(xt,yt)
        mx=beta1*mx+(1-beta1)*dx
        my=beta1*my+(1-beta1)*dy
        
        #vx=beta2*vx+(1-beta2)*dx**2
        #vy=beta2*vy+(1-beta2)*dy**2
        
        ux=max(beta2*ux,np.absolute(dx))
        uy=max(beta2*uy,np.absolute(dy))
        
        if i!=0:
            mmx=1/(1-beta1**i)
            mmy=1/(1-beta1**i)
        else:
            mmx=0
            mmy=0
        #vvx=vx/(1-beta2**i)
        #vvy=vy/(1-beta2**i)
        
        xt=xt-eta*mmx*(mx/ux)
        yt=yt=eta*mmy*(my/uy)
        xs.append(xt)
        ys.append(yt)
    xs=np.asarray(xs)
    ys=np.asarray(ys)
    #print(xs)
    plotPath(xs,ys,"AdaMax")
      
if __name__=="__main__":    
    beta1=0.9
    beta2=0.999
    x_ini,y_ini=-4.4,-4.5   
    epoch=100
    eta=0.2
        
    #run_AdaMax(beta1,beta2,eta,x_ini,y_ini,epoch)
    run_Adam(beta1,beta2,eta,x_ini,y_ini,epoch)        
        
