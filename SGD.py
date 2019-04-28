#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 08:46:09 2019
optimizer:SGD
ref:
    http://cpmarkchang.logdown.com/posts/275500-optimization-method-adagrad
@author: wuzhenglung
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib
import matplotlib.animation as animation

rgn=7
def F(x,y):
    return (-x**2*np.sin(x)+y**2)

def dF(x,y):
    return (-2*x*np.sin(x)-x**2*np.cos(x),2*y)


def plotFx(xt,yt):
    
    fig=plt.figure()
    #https://matplotlib.org/api/_as_gen/matplotlib.pyplot.figure.html
    ax=fig.gca(projection='3d')
    #Get the current Axes instance on the current figure matching the given keyword args, or create one.
    x,y=np.meshgrid(np.arange(-rgn,rgn,0.25),np.arange(-rgn,rgn,0.25))
    #https://docs.scipy.org/doc/numpy/reference/generated/numpy.meshgrid.html
    z=F(x,y)
    surf=ax.plot_surface(x,y,z,cmap=matplotlib.cm.coolwarm)
    ax.scatter(xt, yt, F(xt,yt),c='r', marker='o' )
    ax.set_title("x=%.5f, y=%.5f, f(x,y)=%.5f"%(xt,yt,F(xt,yt))) 
    fig2,ax2=plt.subplots()
    CS = ax2.contourf(x, y, z, cmap=plt.cm.coolwarm)
    #ax2.scatter(xt,yt,c='r',marker='o')
    
    plt.show()
    plt.close()
#rgn=30
def plotPath(xs,ys,title):
    
    fig=plt.figure()
    #https://matplotlib.org/api/_as_gen/matplotlib.pyplot.figure.html
    ax=fig.gca(projection='3d')
    #Get the current Axes instance on the current figure matching the given keyword args, or create one.
    x,y=np.meshgrid(np.arange(-rgn,rgn,0.25),np.arange(-rgn,rgn,0.25))
    #https://docs.scipy.org/doc/numpy/reference/generated/numpy.meshgrid.html
    z=F(x,y)
    surf=ax.plot_surface(x,y,z,cmap=matplotlib.cm.coolwarm)
    ax.scatter(xs, ys, F(xs,ys),c='r', marker='o' )
    ax.set_title("3d surface:"+title) 
    fig2,ax2=plt.subplots()
    plt.plot(xs,ys,'-r')
    CS = ax2.contourf(x, y, z, cmap=plt.cm.coolwarm)
    ax2.set_title("path:"+title) 
    ax2.scatter(xs[0:],ys[0:],c='r', marker='o')
    ax2.scatter(xs[0],ys[0],c='b',marker='o',s=120)
    ax2.scatter(xs[-1],ys[-1],c='gray',marker='o',s=120)
    
    plt.show()
    plt.close()
    
    

def run_SGD(lr,x_ini,y_ini,epoch,weight_decay):
    '''
    lr=learning rate
        
    '''
    c=weight_decay
    xt=x_ini
    yt=y_ini
    
    xs=[]
    ys=[]
    xs.append(xt)
    ys.append(yt)
     
    for i in range(epoch):
        dx,dy=dF(xt,yt)
        #xt=xt-lr*dx
        #yt=yt-lr*dy
        xt=xt-lr*(dx+c*xt)
        yt=yt-lr*(dy+c*yt)
        xs.append(xt)
        ys.append(yt)
    xs=np.asarray(xs)
    ys=np.asarray(ys)
    
    plotPath(xs,ys,"SGD")
        
def run_SGD_Mom(lr,mom,x_ini,y_ini,epoch):
    '''
    lr=learning rate
    mom=momentum
        
    '''
    xt=x_ini
    yt=y_ini
    #plotFx(xt,yt)
    xs=[]
    ys=[]
    xs.append(xt)
    ys.append(yt)
    
    vx=0
    vy=0
     
  
    for i in range(epoch):        
        dx,dy=dF(xt,yt)
        
        vx=mom*vx-lr*dx
        vy=mom*vy-lr*dy
        xt=xt+vx
        yt=yt+vy
        
        xs.append(xt)
        ys.append(yt)
    xs=np.asarray(xs)
    ys=np.asarray(ys)
    #print(xs)
    plotPath(xs,ys,"SGD mom")

def run_NAG(lr,mom,x_ini,y_ini,epoch):
    xt=x_ini
    yt=y_ini
    #plotFx(xt,yt)
    xs=[]
    ys=[]
    xs.append(xt)
    ys.append(yt)
    
    vx=0
    vy=0
       
    for i in range(epoch):        
        dx,dy=dF(xt+mom*vx,yt+mom*vy)
        
        vx=mom*vx-lr*dx
        vy=mom*vy-lr*dy
        xt=xt+vx
        yt=yt+vy
        
        xs.append(xt)
        ys.append(yt)
    xs=np.asarray(xs)
    ys=np.asarray(ys)
    #print(xs)
    plotPath(xs,ys,"Nesterov Accelerated Gradient")

#def run_ASGD(lr,mom,decay,x_ini,y_ini,epoch):
        
    
   
    
if __name__=="__main__":    
    
    plotFx(0,0)
    x_ini,y_ini=-.5,-4.4
    lr=0.01
    mom=0.9
    epoch=100
    weight_decay=0
    run_SGD(lr,x_ini,y_ini,epoch,weight_decay)
    run_SGD_Mom(lr,mom,x_ini,y_ini,epoch)
    run_NAG(lr,mom,x_ini,y_ini,epoch)