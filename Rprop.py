#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 15:01:05 2019
Rprop
@author: wuzhenglung
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib
import matplotlib.animation as animation
import SGD
import math


def run_Rprop(eta_p,eta_n,x_ini,y_ini,epoch):
    xt=x_ini
    yt=y_ini
    
    dx_1=0
    dy_1=0
    
    delta_max=50
    delta_min=1e-6
    
    del_0=0.1
    del_x=del_0
    del_y=del_0
    
    
    xs=[]
    ys=[]
    xs.append(xt)
    ys.append(yt)
    
    
    def delta(d_1,d,delta_a):
        r=d_1*d
        if r>0:
            delta_a=min(eta_p*delta_a,delta_max)
        elif r<0:
            delta_a=max(eta_n*delta_a,delta_min)
        else:
            delta_a=delta_a
        return delta_a
    
    def update(d_1,d,w):
        r=d_1*d
        if r>=0:
            w=-sign(d)*w
        else:
            w=-w
            d=0
        return w,d
                        
    def sign(val):
        if val>0:
            val=1
        elif val<0:
            val=-1
        return val
    
    
    
    for i in range(epoch):
        dx,dy=dF(xt,yt)
        
        del_x=delta(dx_1,dx,del_x)        
        del_y=delta(dy_1,dy,del_y)

        deltax,dx_1=update(dx_1,dx,del_x)
        deltay,dy_1=update(dy_1,dy,del_y)
        
        xt=xt+deltax
        yt=yt+deltay
        
        xs.append(xt)
        ys.append(yt)
        
        
    xs=np.asarray(xs)
    ys=np.asarray(ys)
    #print(xs)
    plotPath(xs,ys,"Rprop")
        
       
        
        
    

            
if __name__=="__main__":    
    
    x_ini,y_ini=-4.5,-4.4   
    epoch=100
    eta_p=-0.5
    eta_n=0.4        
    run_Rprop(eta_p,eta_n,x_ini,y_ini,epoch)    
        