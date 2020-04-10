#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 17:33:14 2020

@author: basile
"""




class animTrebuchet:
    def __init__(self, Zsol, par):
        self.Zsol=Zsol
        self.par=par
        self.fig, self.ax = plt.subplots()
        self.ln, = self.ax.plot([], [])
        self.frames = range(0, np.shape(Zsol)[0])
        self.ax.set_aspect(aspect='equal', adjustable='box')
        self.ax.set_xlim(left=-3, right=3)
        self.ax.set_ylim(bottom=-0.5, top=5)
        self.ax.set_xbound(lower=-5, upper=5)
        self.ax.set_ybound(lower=-0.5, upper=5)
        self.ax.grid(b=True)
        # geometric stuff
        self.axle = plt.Circle([0,0], radius=0.07)
        self.m2 = plt.Circle([0,0], radius=0.3,color=[1.0,0.2,0.2,1.0])
        self.joint = plt.Circle([0,0], radius=0.07)
        self.m3 = plt.Circle([0,0], radius=0.1)
        self.ax.add_patch(self.axle)
        self.ax.add_patch(self.joint)
        self.ax.add_patch(self.m2)
        self.ax.add_patch(self.m3)
        
    def updateTreb(self, frame):
        z = self.Zsol[frame,:3]
        p1 = p1_f(z, self.par)
        p2 = p2_f(z, self.par)
        pBar = pBar_f(z, self.par)
        p3 = p3_f(z, self.par)
        x = np.concatenate((p2,pBar,p3), axis=1)   
        self.axle.set_center(p1)
        self.m2.set_center(p2)
        self.joint.set_center(pBar)
        self.m3.set_center(p3)
        self.ln.set_xdata(x[0,:])
        self.ln.set_ydata(x[1,:])
       
    def anim(self):
        return FuncAnimation(self.fig, self.updateTreb, self.frames, blit=False, repeat_delay=1000, interval=50)
        

treb = animTrebuchet(Zsol, parNum0)
aa = treb.anim()

trebrev = animTrebuchet(Zsol[29:0:-1,:], parNum0)
aarev = trebrev.anim()
        