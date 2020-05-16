#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimise energy transfer of a trebuchet with bearing.
Created on Sat Apr  4 16:27:39 2020

@author: basile
"""

import sympy as sp
import numpy as np
import scipy.integrate as integrate
import scipy.optimize as opt
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

# Use qt for animations
try:
    import IPython
    shell = IPython.get_ipython()
    shell.enable_matplotlib(gui='qt')
except:
    pass


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
parNum0 = np.array([10.0,150.0,2.0,9.81,2.0,3.0,3.0])


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
# mass * ztt + coriolisAndPot + lam * dcdz == [0,0,0]
# lam >= 0
dLdzt = sp.Matrix([L]).jacobian(zt).transpose()
dLdz = sp.Matrix([L]).jacobian(z).transpose()

mass = dLdzt.jacobian(zt)
coriolisAndPot = dLdzt.jacobian(z) * zt - dLdz
dcdz = sp.Matrix([c]).jacobian(z).transpose()

# after launch
dLdztFree = sp.Matrix([LFree]).jacobian(zt[:2,:]).transpose()
dLdzFree = sp.Matrix([LFree]).jacobian(z[:2,:]).transpose()

massFree = dLdztFree.jacobian(zt[:2,:])
coriolisAndPotFree = dLdztFree.jacobian(z[:2,:]) * zt[:2,:] - dLdzFree


# expressions for constraints derivative 
ddc_dzdz = dcdz.jacobian(z)

# make functions
mass_f = sp.lambdify(((z),(par),), mass)
coriolisAndPot_f = sp.lambdify(((z),(zt),(par),), coriolisAndPot)
dcdz_f = sp.lambdify(((z),(par),), dcdz)
ddc_dzdz_f = sp.lambdify(((z),(par),), ddc_dzdz)

Etot_f = sp.lambdify(((z),(zt),(par),), Etot)
Ekm3_f = sp.lambdify(((z),(zt),(par),), Ekm3)

massFree_f = sp.lambdify(((z[:2,:]),(par),), massFree)
coriolisAndPotFree_f = sp.lambdify(((z[:2,:]),(zt[:2,:]),(par),), coriolisAndPotFree)

# initial solution; th2 as a function of th1 such that m3 is on the floor
th2Init = sp.asin(-(h + la * sp.sin(th1))/lb)
th2Init_f = sp.lambdify(((th1),(par),), th2Init)
c_f = sp.lambdify(((z),(par),), c)


def solve_lam(z, zt, par):
    m_N = mass_f(z, par)
    mInv_N = np.linalg.inv(m_N)
    dcdz_N = dcdz_f(z, par)
    ddc_dzdz_N = ddc_dzdz_f(z, par)
    coriolisAndPot_N = coriolisAndPot_f(z, zt, par)
    den = np.matmul(np.matmul(dcdz_N.transpose(), mInv_N), dcdz_N)
    num1 = np.matmul(np.matmul(zt, ddc_dzdz_N), zt)
    num2 = np.matmul(np.matmul(dcdz_N.transpose(), mInv_N), coriolisAndPot_N)
    lam_N = (num1 - num2) / den
    return lam_N, m_N, coriolisAndPot_N, dcdz_N


def drift(z, zt, par):
    lam_N, m_N, coriolisAndPot_N, dcdz_N = solve_lam(z, zt, par)
    c_N = c_f(z, par)
    if (lam_N <= 0) and (c_N < 1e-5):
        # floor constraint active
        ztt = np.linalg.solve(m_N, - coriolisAndPot_N - lam_N * dcdz_N)
    else:
        # floor constraint inactive
        ztt = np.linalg.solve(m_N, - coriolisAndPot_N)
    return ztt[:,0]

def drift1(Z,par):
    z = Z[:3]
    zt = Z[3:]
    ztt = drift(z, zt, par);
    Zt = np.concatenate((zt, ztt))
    return Zt

def driftFree(z, zt, par):
    mF_N = massFree_f(z, par)
    coriolisAndPotFree_N = coriolisAndPotFree_f(z, zt, par)
    ztt = np.linalg.solve(mF_N, - coriolisAndPotFree_N)
    return ztt[:,0]

def drift1Free(Z,par):
    z = Z[:2]
    zt = Z[2:]
    ztt = driftFree(z, zt, par);
    Zt = np.concatenate((zt, ztt))
    return Zt

# stop event function: stop simulation when exit speed vector angle is 45 deg
p3t_f = sp.lambdify(((z),(zt),(par),), p3t)
vvv=[]
def launchEvent(tt, Z, par, tMin = 0.1):
    z = Z[:3]
    zt = Z[3:]
    p3t_N = p3t_f(z, zt, par)
    # Apply small angle to avoid the jump of atan2 at -pi
    v = p3t_N
    phi = -0.01;
    R = np.array([[np.cos(phi), -np.sin(phi)],[np.sin(phi), np.cos(phi)]])
    v = np.matmul(R,v)
    # Avoid 0 vector v at begin and correct for phi rotation
    ev = np.arctan2(v[1,0],v[0,0]-0.000001) - phi - np.arctan2(1,1)
    vvv.append(v)
    return ev 
    

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
tSpan = [0, 5]
tSpanFree = [0, 1]
th1Init = -1.6


# simulation
def simulate(par):
    Z0 = initCond1(th1Init, par)
    fInt = lambda tt, ZZ : drift1(ZZ,par)
    fEvent = lambda tt, ZZ : launchEvent(tt, ZZ,par)
    fEvent.terminal = True
    ivpSol = integrate.solve_ivp(fInt, tSpan, Z0, events = fEvent, rtol=1.0e-5, atol=1.0e-8, max_step = 0.01)
    # Free
    if ivpSol.success & ivpSol.status == 1:
        ZFree0 = ivpSol.y[[0,1,3,4],-1]
        fFreeInt = lambda tt, ZZ : drift1Free(ZZ,par)
        ivpSolFree = integrate.solve_ivp(fFreeInt, tSpanFree, ZFree0, rtol=1.0e-5, atol=1.0e-8)
        return ivpSol, ivpSolFree
    else:
        return ivpSol, []

ivpSol, ivpSolFree = simulate(parNum0)


# evaluate enery transfer efficiency
def efficiency(par):
    try:
        ivpSol, ivpSolFree = simulate(par)          
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
    

ivpSol, ivpSolFree = simulate(parNum0)
ee=[Etot_f(zz[:3],zz[3:],parNum0) for zz in ivpSol.y.transpose()]
plt.figure()
plt.plot(ee)
plt.show()



# Reinterpolate both solutions at constant time steps
def reinterp(ivpSol, ivpSolFree):
    tLaunch = ivpSol.t[-1]
    tEnd = tLaunch + ivpSolFree.t[-1]
    t = np.linspace(0, tEnd, 100)
    y = ivpSol.y
    yF = ivpSolFree.y
    yF = np.concatenate((yF[:2,:],np.zeros((1,yF.shape[1])), yF[2:,:],np.zeros((1,yF.shape[1])) ))
    yAll = np.concatenate((y,yF[:,1:]), axis=1)
    ty = ivpSol.t
    tyF = ivpSolFree.t + tLaunch
    tyAll = np.concatenate((ty,tyF[1:]))
    Zsol = np.array([np.interp(t, tyAll, yAll[k,:]) for k in range(yAll.shape[0])])
    Zsol = Zsol.transpose()
    kLast = np.where(t<tLaunch)[0][-1]
    return t, Zsol, kLast

t, Zsol, kLast = reinterp(ivpSol, ivpSolFree)

plt.plot(t, Zsol)

    

# functions for animation
p1_f=sp.lambdify(((z),(par),), p1)
p2_f=sp.lambdify(((z),(par),), p2)
pBar_f=sp.lambdify(((z),(par),), pBar)
p3_f=sp.lambdify(((z),(par),), p3)


class animTrebuchet:
    def __init__(self, Zsol, kLast, par, ivpSol):
        self.Zsol=Zsol
        self.kLast = kLast
        self.par=par
        self.fig, self.ax = plt.subplots()
        self.frames = range(0, np.shape(Zsol)[0])
        # non resampled m3 path
        self.m3Path = np.array(
                [p3_f(ivpSol.y.transpose()[k][:3], par)[:,0] 
                for k in range(0,ivpSol.y.shape[1])]
                )
        
    def initAnim(self):
        self.ax.clear()
        self.ln, = self.ax.plot([], [])
        self.ax.set_aspect(aspect='equal', adjustable='box')
        self.ax.set_xlim(left=-3, right=3)
        self.ax.set_ylim(bottom=-0.5, top=8)
        self.ax.set_xbound(lower=-5, upper=5)
        self.ax.set_ybound(lower=-0.5, upper=8)
        self.ax.grid(b=True)
        # geometric stuff
        hh = self.par[2]
        self.ax.plot([-2,2],[hh,hh], color=[0.6,0.6,0.6])
        self.axle = plt.Circle([0,0], radius=0.1)
        self.m2 = plt.Circle([0,0], radius=0.3,color=[1.0,0.2,0.2,1.0])
        self.joint = plt.Circle([0,0], radius=0.05)
        self.m3 = plt.Circle([0,0], radius=0.15)
        self.ax.add_patch(self.axle)
        self.ax.add_patch(self.joint)
        self.ax.add_patch(self.m2)
        self.ax.add_patch(self.m3)
        # m3 path
        self.ax.plot(self.m3Path[:,0], self.m3Path[:,1], color=[0.6,0.9,0.9], zorder=-1)
        
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
        if frame <= self.kLast:
            self.ln.set_xdata(x[0,:])
            self.ln.set_ydata(x[1,:])
            self.m3.set_radius(0.1)
            mm3 = plt.Circle(p3, radius=0.05, color=[1,0,0])
            self.ax.add_patch(mm3)
        else:
            self.ln.set_xdata(x[0,:-1])
            self.ln.set_ydata(x[1,:-1])
            self.m3.set_radius(0)
       
    def anim(self):
        return FuncAnimation(self.fig, self.updateTreb, self.frames, init_func=self.initAnim, blit=False, repeat_delay=1000, interval=50)
        

treb = animTrebuchet(Zsol, kLast, parNum0, ivpSol)
aa = treb.anim()

    
    
print(efficiency(parNum0))

# optimise
if True:  
    pparOpt0 = parNum0[5:]
    hh = parNum0[2]
    mm3 = parNum0[6]
    parOptLB = [0.6*hh, 0.7*mm3]
    parOptUB = [2.0*hh, 1.3*mm3]
    bnds = opt.Bounds(parOptLB, parOptUB)
    fMin = lambda pO : -efficiency(np.concatenate((parNum0[0:5], pO)))
    optSol = opt.minimize(fMin, pparOpt0, bounds=bnds, options={'disp': True})
    
    parOpt = np.concatenate((parNum0[0:5], optSol.x))
    ivpOpt, ivpOptFree = simulate(parOpt)
    
    tOpt, ZsolOpt, kLastOpt = reinterp(ivpOpt, ivpOptFree)
  
    trebOpt = animTrebuchet(ZsolOpt, kLastOpt, parOpt, ivpOpt)
    aaOpt = trebOpt.anim()
    
    print(efficiency(parNum0))
    print(efficiency(parOpt))
    
    
    




