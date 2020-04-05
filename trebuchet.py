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

pFixed = sp.Matrix([m1,m2,h,g])
pFixedNum = np.array([10.0,100.0,2.0,9.81])

pOptim = sp.Matrix([la,lb,m3])
pOptimNum0 = np.array([2.0,2.0,1.0])
#pOptimNum0 = np.array([2.0,2.99,3.0])


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
mass_f = sp.lambdify(((z),(pFixed),(pOptim),), mass)
coriolis_f = sp.lambdify(((z),(zt),(pFixed),(pOptim),), coriolis)
dcdz_f = sp.lambdify(((z),(pFixed),(pOptim),), dcdz)
ddc_dzdz_f = sp.lambdify(((z),(pFixed),(pOptim),), ddc_dzdz)

Etot_f = sp.lambdify(((z),(zt),(pFixed),(pOptim),), Etot)
Ekm3_f = sp.lambdify(((z),(zt),(pFixed),(pOptim),), Ekm3)

massFree_f = sp.lambdify(((z[:2,:]),(pFixed),(pOptim),), massFree)
coriolisFree_f = sp.lambdify(((z[:2,:]),(zt[:2,:]),(pFixed),(pOptim),), coriolisFree)

# initial solution; th2 as a function of th1 such that m3 is on the floor
th2Init = sp.asin(-(h + la * sp.sin(th1))/lb)
th2Init_f = sp.lambdify(((th1),(pFixed),(pOptim),), th2Init)
c_f = sp.lambdify(((z),(pFixed),(pOptim),), c)


def solve_lam(z, zt, pFixed, pOptim):
    m_N = mass_f(z, pFixed, pOptim)
    mInv_N = np.linalg.inv(m_N)
    dcdz_N = dcdz_f(z, pFixed, pOptim)
    ddc_dzdz_N = ddc_dzdz_f(z, pFixed, pOptim)
    coriolis_N = coriolis_f(z, zt, pFixed, pOptim)
    den = np.matmul(np.matmul(dcdz_N.transpose(), mInv_N), dcdz_N)
    num1 = np.matmul(np.matmul(zt, ddc_dzdz_N), zt)
    num2 = np.matmul(np.matmul(dcdz_N.transpose(), mInv_N), coriolis_N)
    lam_N = (num1 - num2) / den
    return lam_N, m_N, coriolis_N, dcdz_N


def drift(z, zt, pFixed, pOptim):
    lam_N, m_N, coriolis_N, dcdz_N = solve_lam(z, zt, pFixed, pOptim)
    c_N = c_f(z, pFixed, pOptim)
    if (lam_N <= 0) and (c_N < 1e-5):
        # floor constraint active
        ztt = np.linalg.solve(m_N, - coriolis_N - lam_N * dcdz_N)
    else:
        # floor constraint inactive
        ztt = np.linalg.solve(m_N, - coriolis_N)
    return ztt[:,0]

def drift1(Z,pFixed, pOptim):
    z = Z[:3]
    zt = Z[3:]
    ztt = drift(z, zt, pFixed, pOptim);
    Zt = np.concatenate((zt, ztt))
    return Zt

# stop event function: stop simulation when exit speed vector angle is 45 deg
p3t_f = sp.lambdify(((z),(zt),(pFixed),(pOptim),), p3t)
def launchEvent(Z, pFixed, pOptim):
    z = Z[:3]
    zt = Z[3:]
    p3t_N = p3t_f(z, zt, pFixed, pOptim)
    # avoid division by 0 at begin
    v = p3t_N + np.array([[0.0], [1.0e-3]])
    return v[0,0]/v[1,0] - 1.0
    

# initial conditions
def initCond(th10, pFixed, pOptim):
    th20 = th2Init_f(th10, pFixed, pOptim)
    assert(np.all(np.isreal(th20)))
    if np.cos(th20) < 0:
        th20 = np.pi - th20
    z0 = np.array([0,th10,th20])
    zt0 = np.array([0.0,0.0,0.0])   
    return z0, zt0

def initCond1(th10, pFixed, pOptim):
    z0, zt0 = initCond(th10, pFixed, pOptim)
    return np.concatenate((z0, zt0))


# simulation paramters
tSpan = [0, 1.2]
th1Init = -1.8


# simulation
def simulate(pFixed, pOptim):
    Z0 = initCond1(th1Init, pFixed, pOptim)
    fInt = lambda tt, ZZ : drift1(ZZ,pFixed, pOptim)
    fEvent = lambda tt, ZZ : launchEvent(ZZ,pFixed, pOptim)
    fEvent.terminal = True
    ivpSol = integrate.solve_ivp(fInt, tSpan, Z0, events = fEvent, rtol=1.0e-6, atol=1.0e-9)
    return ivpSol

ivpSol = simulate(pFixedNum, pOptimNum0)


# evaluate enery transfer efficiency
def efficiency(pFixed, pOptim):
    ivpSol = simulate(pFixed, pOptim)
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
        Etot0_N = Etot_f(z0, zt0, pFixed, pOptim)
        Ekm3F_N = Ekm3_f(zF, ztF, pFixed, pOptim)
        return Ekm3F_N / Etot0_N
    else:
        assert(False)
        return 0.0
    

ivpSol = simulate(pFixedNum, pOptimNum0)
ee=[Etot_f(zz[:3],zz[3:],pFixedNum, pOptimNum0) for zz in ivpSol.y.transpose()]
plt.figure()
plt.plot(ee)
plt.show()

# Reinterpolate at constant time steps
t = np.linspace(0, ivpSol.t[-1], 30)
Zsol = np.array([np.interp(t, ivpSol.t, ivpSol.y[k,:]) for k in range(ivpSol.y.shape[0])])
Zsol = Zsol.transpose()

plt.plot(t, Zsol)
    

# functions for animation
p1_f=sp.lambdify(((z),(pFixed),(pOptim),), p1)
p2_f=sp.lambdify(((z),(pFixed),(pOptim),), p2)
pBar_f=sp.lambdify(((z),(pFixed),(pOptim),), pBar)
p3_f=sp.lambdify(((z),(pFixed),(pOptim),), p3)

fig, ax = plt.subplots()
ln, = plt.plot([], [], 'ro')
frames = range(0,len(t))
def init():
    ax.clear()   
    ax.set_aspect(aspect='equal', adjustable='box')
    ax.set_xlim(left=-3, right=3)
    ax.set_ylim(bottom=-0.5, top=5)
    ax.set_xbound(lower=-5, upper=5)
    ax.set_ybound(lower=-0.5, upper=5)
    plt.grid(b=True)
    return ln,

def showTrebuchet(z, pFixed, pOptim, ax):
    z = z[:3]
    p1 = p1_f(z, pFixed, pOptim)
    p2 = p2_f(z, pFixed, pOptim)
    pBar = pBar_f(z, pFixed, pOptim)
    p3 = p3_f(z, pFixed, pOptim)
    x = np.concatenate((p2,pBar,p3), axis=1)    
    init()
    axle = plt.Circle(p1, radius=0.07)
    m2 = plt.Circle(p2, radius=0.3,color=[1.0,0.2,0.2,1.0])
    joint = plt.Circle(pBar, radius=0.07)
    m3 = plt.Circle(p3, radius=0.1)
    ax.add_patch(axle)
    ax.add_patch(joint)
    ax.add_patch(m2)
    ax.add_patch(m3)
    plt.plot(x[0,:],x[1,:])

    
    
    
update = lambda k : showTrebuchet(Zsol[k,:], pFixedNum, pOptimNum0, ax)

    
ani = FuncAnimation(fig, update, frames,
                    init_func=init, blit=False)
plt.show()










