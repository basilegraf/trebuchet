#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimise enery transfer of a wheeled trebuchet.
Created on Sat Apr  4 16:27:39 2020

@author: basile
"""

import sympy as sp
import numpy as np
import scipy.integrate as integrate
import scipy.optimize as opt
import matplotlib.pyplot as plt


import matplotlib.path as mpath
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

from matplotlib.animation import FuncAnimation

plt.close('all')

# state variables and derivatives
x = sp.symbols('x')
x, th1, th2 = sp.symbols('x, th1, th2')
xt, th1t, th2t = sp.symbols('xt, th1t, th2t')
xtt, th1tt, th2tt = sp.symbols('xtt, th1tt, th2tt')
# fixed parameters
m1,m2,h,g = sp.symbols('m1,m2,h,g')
# optimised parameters
la,lb,m3 = sp.symbols('la,lb,m3')
# Lagrange multiplier for floor constraint
lam = sp.symbols('lam')

par = sp.Matrix([m1,m2,h,g,la,lb,m3])
parNum0 = np.array([1.0,100.0,2.0,9.81,2.0,3.0,3.0])


# state vector and derivatives
z = sp.Matrix([x,th1,th2])
zt = sp.Matrix([xt,th1t,th2t])
ztt = sp.Matrix([xtt,th1tt,th2tt])

# positions
p1=sp.Matrix([x, h])
p2 = p1 - h * sp.Matrix([sp.cos(th1),sp.sin(th1)])
pBar = p1 + la * sp.Matrix([sp.cos(th1),sp.sin(th1)])
p3 = pBar + lb * sp.Matrix([sp.cos(th2),sp.sin(th2)])

# speeds
p1t = p1.jacobian(z) * zt 
p2t = p2.jacobian(z) * zt
p3t = p3.jacobian(z) * zt

# kinetic energy
Ekm1 = (m1/2) * p1t.dot(p1t)
Ekm2 = (m2/2) * p2t.dot(p2t)
Ekm3 = (m3/2) * p3t.dot(p3t)
Ekin = Ekm1 + Ekm2 + Ekm3
# after launch
EkinFree = Ekm1 + Ekm2

# potential energy
ey = sp.Matrix([0,1])
Epm2 = m2 * g * ey.dot(p2)
Epm3 = m3 * g * ey.dot(p3)
Epot = Epm2 + Epm3
# after launch
EpotFree = Epm2

# Lagrangian
L = Ekin - Epot
LFree = EkinFree - EpotFree

# total energy
Etot = Ekin + Epot

# floor constraint for mass point p3
c = ey.dot(p3)


# equations of motion:
# mass * ztt + coriolis + lam * dcdz == [0,0,0]
# lam >= 0
dLdzt = sp.Matrix([L]).jacobian(zt).transpose()
dLdz = sp.Matrix([L]).jacobian(z).transpose()

mass = dLdzt.jacobian(zt)
coriolis = dLdzt.jacobian(z) * zt - dLdz
dcdz = sp.Matrix([c]).jacobian(z).transpose()

# after launch
dLdztFree = sp.Matrix([LFree]).jacobian(zt[:2,:]).transpose()
dLdzFree = sp.Matrix([LFree]).jacobian(z[:2,:]).transpose()

massFree = dLdztFree.jacobian(zt[:2,:])
coriolisFree = dLdztFree.jacobian(z[:2,:]) * zt[:2,:] - dLdzFree


# expressions for constraints derivative 
ddc_dzdz = dcdz.jacobian(z)

# make functions
mass_f = sp.lambdify(((z),(par),), mass)
coriolis_f = sp.lambdify(((z),(zt),(par),), coriolis)
dcdz_f = sp.lambdify(((z),(par),), dcdz)
ddc_dzdz_f = sp.lambdify(((z),(par),), ddc_dzdz)

Etot_f = sp.lambdify(((z),(zt),(par),), Etot)
Ekm3_f = sp.lambdify(((z),(zt),(par),), Ekm3)

massFree_f = sp.lambdify(((z[:2,:]),(par),), massFree)
coriolisFree_f = sp.lambdify(((z[:2,:]),(zt[:2,:]),(par),), coriolisFree)

# initial solution; th2 as a function of th1 such that m3 is on the floor
th2Init = sp.asin(-(h + la * sp.sin(th1))/lb)
th2Init_f = sp.lambdify(((th1),(par),), th2Init)
c_f = sp.lambdify(((z),(par),), c)


def solve_lam(z, zt, par):
    m_N = mass_f(z, par)
    mInv_N = np.linalg.inv(m_N)
    dcdz_N = dcdz_f(z, par)
    ddc_dzdz_N = ddc_dzdz_f(z, par)
    coriolis_N = coriolis_f(z, zt, par)
    den = np.matmul(np.matmul(dcdz_N.transpose(), mInv_N), dcdz_N)
    num1 = np.matmul(np.matmul(zt, ddc_dzdz_N), zt)
    num2 = np.matmul(np.matmul(dcdz_N.transpose(), mInv_N), coriolis_N)
    lam_N = (num1 - num2) / den
    return lam_N, m_N, coriolis_N, dcdz_N


def drift(z, zt, par):
    lam_N, m_N, coriolis_N, dcdz_N = solve_lam(z, zt, par)
    c_N = c_f(z, par)
    if (lam_N <= 0) and (c_N < 1e-5):
        # floor constraint active
        ztt = np.linalg.solve(m_N, - coriolis_N - lam_N * dcdz_N)
    else:
        # floor constraint inactive
        ztt = np.linalg.solve(m_N, - coriolis_N)
    return ztt[:,0]

def drift1(Z,par):
    z = Z[:3]
    zt = Z[3:]
    ztt = drift(z, zt, par);
    Zt = np.concatenate((zt, ztt))
    return Zt

# stop event function: stop simulation when exit speed vector angle is 45 deg
p3t_f = sp.lambdify(((z),(zt),(par),), p3t)
def launchEvent(tt, Z, par, tMin = 0.1):
    z = Z[:3]
    zt = Z[3:]
    p3t_N = p3t_f(z, zt, par)
    # avoid division by 0 at begin
    v = p3t_N + np.array([[1.0e-5], [0.5e-5]])
    return v[1,0]/v[0,0] - 1.0 
    

# initial conditions
def initCond(th10, par):
    th20 = th2Init_f(th10, par)
    assert(np.all(np.isreal(th20)))
    if np.cos(th20) < 0:
        th20 = np.pi - th20
    z0 = np.array([0,th10,th20])
    zt0 = np.array([0.0,0.0,0.0])   
    return z0, zt0

def initCond1(th10, par):
    z0, zt0 = initCond(th10, par)
    return np.concatenate((z0, zt0))


# simulation paramters
tSpan = [0, 1.2]
th1Init = -1.8


# simulation
def simulate(par):
    Z0 = initCond1(th1Init, par)
    fInt = lambda tt, ZZ : drift1(ZZ,par)
    fEvent = lambda tt, ZZ : launchEvent(tt, ZZ,par)
    fEvent.terminal = True
    ivpSol = integrate.solve_ivp(fInt, tSpan, Z0, events = fEvent, rtol=1.0e-5, atol=1.0e-8)
    return ivpSol

ivpSol = simulate(parNum0)


# evaluate enery transfer efficiency
def efficiency(par):
    try:
        ivpSol = simulate(par)          
        if not(ivpSol.success):
            return 0.0
        elif ivpSol.status == -1:
            # integration step failed
            return 0.0
        elif ivpSol.status == 0:
            # final time reached, no launch at all
            return 0.0
        elif ivpSol.status == 1:
            # launch event reached, compute energy transfert
            z0 = ivpSol.y[:3,0]
            zt0 = ivpSol.y[3:,0]
            zF = ivpSol.y[:3,-1]
            ztF = ivpSol.y[3:,-1]
            Etot0_N = Etot_f(z0, zt0, par)
            Ekm3F_N = Ekm3_f(zF, ztF, par)
            return Ekm3F_N / Etot0_N
        else:
            assert(False)
            return 0.0
    except:
        return 0.0
    

ivpSol = simulate(parNum0)
ee=[Etot_f(zz[:3],zz[3:],parNum0) for zz in ivpSol.y.transpose()]
plt.figure()
plt.plot(ee)
plt.show()

# Reinterpolate at constant time steps
t = np.linspace(0, ivpSol.t[-1], 30)
Zsol = np.array([np.interp(t, ivpSol.t, ivpSol.y[k,:]) for k in range(ivpSol.y.shape[0])])
Zsol = Zsol.transpose()

plt.plot(t, Zsol)
    

# functions for animation
p1_f=sp.lambdify(((z),(par),), p1)
p2_f=sp.lambdify(((z),(par),), p2)
pBar_f=sp.lambdify(((z),(par),), pBar)
p3_f=sp.lambdify(((z),(par),), p3)


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

    
    



if False:  
    pparOpt0 = parNum0[4:]
    parOptLB = 0.3 * parNum0[4:]
    parOptUB = 3.0 * parNum0[4:]
    bnds = opt.Bounds(parOptLB, parOptUB)
    fMin = lambda pO : -efficiency(np.concatenate((parNum0[0:4], pO)))
    optSol = opt.minimize(fMin, pparOpt0, bounds=bnds, options={'disp': True})
    
    parOpt = np.concatenate((parNum0[0:4], optSol.x))
    ivpOpt = simulate(parOpt)
    
    ZsolOpt = np.array([np.interp(t, ivpOpt.t, ivpOpt.y[k,:]) for k in range(ivpOpt.y.shape[0])])
    ZsolOpt = ZsolOpt.transpose()
  
    fig, ax = plt.subplots()
    ln, = plt.plot([], [], 'ro')
    update = lambda k : showTrebuchet(ZsolOpt[k,:], parNum0, ax)
    aniOpt = FuncAnimation(fig, update, frames,init_func=init, blit=False)
    plt.show()
    
    #method='SLSQP',
    
    



