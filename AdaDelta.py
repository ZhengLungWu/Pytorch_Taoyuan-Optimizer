#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 11:46:03 2019
AdaDelta
RMSprop
ref:
    http://cpmarkchang.logdown.com/posts/467674-optimization-method-adadelta
@author: wuzhenglung
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib
import matplotlib.animation as animation
import SGD



def run_AdaDelta(rho,x_ini,y_ini,epoch):
    
    eps=1e-2
    xt=x_ini
    yt=y_ini
    #plotFx(xt,yt)
    xs=[]
    ys=[]
    xs.append(xt)
    ys.append(yt)
    Egxti=0
    Egyti=0
    Ex=0
    Ey=0
    
     
    for i in range(epoch):
        dx,dy=dF(xt,yt)
        
        Egxti=rho*Egxti+(1-rho)*dx**2
        Egyti=rho*Egyti+(1-rho)*dy**2
        RMSgx=(Egxti+eps)**0.5
        RMSgy=(Egyti+eps)**0.5
        
        RMSEx_1=(Ex+eps)**0.5
        RMSEy_1=(Ey+eps)**0.5
        deltaX=-(RMSEx_1/RMSgx)*dx
        deltaY=-(RMSEy_1/RMSgy)*dy
        xt=xt+deltaX
        yt=yt+deltaY
        Ex=rho*Ex+(1-rho)*deltaX**2
        Ey=rho*Ey+(1-rho)*deltaY**2
        
        xs.append(xt)
        ys.append(yt)
    xs=np.asarray(xs)
    ys=np.asarray(ys)
    #print(xs)
    plotPath(xs,ys,"AdaDelta")

def run_RMSprop(rho,eta,x_ini,y_ini,epoch):
    eps=1e-8
    xt=x_ini
    yt=y_ini
    #plotFx(xt,yt)
    xs=[]
    ys=[]
    xs.append(xt)
    ys.append(yt)
    Egxti=0
    Egyti=0
    #Ex=0
    #Ey=0
    
     
    for i in range(epoch):
        dx,dy=dF(xt,yt)
        
        Egxti=rho*Egxti+(1-rho)*dx**2
        Egyti=rho*Egyti+(1-rho)*dy**2
        RMSgx=(Egxti+eps)**0.5
        RMSgy=(Egyti+eps)**0.5
        
        #RMSEx_1=(Ex+eps)**0.5
        #RMSEy_1=(Ey+eps)**0.5
        deltaX=-(eta/RMSgx)*dx
        deltaY=-(eta/RMSgy)*dy
        xt=xt+deltaX
        yt=yt+deltaY
        #Ex=rho*Ex+(1-rho)*deltaX**2
        #Ey=rho*Ey+(1-rho)*deltaY**2
        
        xs.append(xt)
        ys.append(yt)
    xs=np.asarray(xs)
    ys=np.asarray(ys)
    #print(xs)
    plotPath(xs,ys,"RMSprop")
    
if __name__=="__main__":        
    rho=0.9
    eta=0.1
    epoch=100
    x_ini,y_ini=-4.5,-4.4
    #run_AdaDelta(rho,x_ini,y_ini,epoch)
    run_RMSprop(rho,eta,x_ini,y_ini,epoch)