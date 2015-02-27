#!/usr/bin/env python
from numpy import *
from numpy.linalg import *
import scipy.optimize
from matplotlib.pyplot import *
from time import time
from scipy.optimize import fsolve
from sys import exit


# Unteraufgabe a)

def rhs(y, D=3):
    """Right hand side

    Input: y ... array of q and p:  y = (q, p)
           D ...  space dimension, default equals 3.

    Output: dy ... time-derivative of y
    """
    q, p = hsplit(y, 2)
    # Number of bodies
    N = q.size // D
    # Empty arrays for computed data
    dq = zeros_like(p)
    dp = zeros_like(q)
    #######################################################
    #                                                     #
    # TODO: Implementieren Sie hier die rechte Seite f(y) #
    #       wobei y = (q, p) wie in der Beschreibung.     #
    #                                                     #
    #######################################################

    for k in range(N):
        for i in range(D):
            dq[(k*D)+i] = p[(k*D)+i] / m[k]

    for k in range(N):
        for i in range(N):
            if i != k:
                qi = q[i*D:(i*D+D)]
                qk = q[k*D:(k*D+D)]
                dp[k*D:(k*D+D)] += G*(m[k]*m[i]) / (norm(qi-qk)**3) * (qi - qk)

    dy = hstack([dq, dp])
    return dy


# Unteraufgabe c)

def rhs_vv(q, D=3):
    """Right hand side

    Input: q ...  array with positions

    Output: dv ... time-derivative of the velocities
    """
    # Number of bodies
    N = q.size // D
    # Empty arrays for computed data
    dp = zeros_like(q)
    #######################################################
    #                                                     #
    # TODO: Implementieren Sie hier die rechte Seite f(y) #
    #       geeignet fuer die velocity Verlet Methode.    #
    #                                                     #
    #######################################################
    for k in range(N):
        for i in range(N):
            if i != k:
                qi = q[i*D:(i*D+D)]
                qk = q[k*D:(k*D+D)]
                dp[k*D:(k*D+D)] += G*(m[k]*m[i]) / (norm(qi-qk)**3) * (qi - qk)

    dv = (dp.reshape((N,D)) / m[:,newaxis]).flatten()
    return dv


# Unteraufgabe b)

def integrate_EE(y0, xStart, xEnd, steps, flag=False):
    r"""Integrate ODE with explicit Euler method

    Input: y0     ... initial condition
           xStart ... start x
           xEnd   ... end   x
           steps  ... number of steps (h = (xEnd - xStart)/N)
           flag   ... flag == False return complete solution: (phi, phi', t)
                      flag == True  return solution at endtime only: phi(tEnd)

    Output: x ... variable
            y ... solution
    """
    x = zeros(steps)
    y = zeros((steps, size(y0)))
    ###########################################################
    #                                                         #
    # TODO: Implementieren Sie hier die explizite Euler Regel #
    #       zur integration der funktion y(x).                #
    #                                                         #
    ###########################################################

    h = double(xEnd)/steps
    y[0] = y0

    for k in xrange(steps-1):
        y[k+1] = y[k] + h * rhs(y[k])
        x[k+1] = (k+1)*h

    if flag:
        return x[-1], y[-1][:]
    else:
        return x, y


def integrate_IE(y0, xStart, xEnd, steps, flag=False):
    r"""Integrate ODE with implicit Euler method

    Input: y0     ... initial condition
           xStart ... start x
           xEnd   ... end   x
           steps  ... number of steps (h = (xEnd - xStart)/N)
           flag   ... flag == False return complete solution: (phi, phi', t)
                      flag == True  return solution at endtime only: phi(tEnd)

    Output: x ... variable
            y ... solution
    """
    x = zeros(steps)
    y = zeros((steps, size(y0)))
    ###########################################################
    #                                                         #
    # TODO: Implementieren Sie hier die implizite Euler Regel #
    #       zur integration der funktion y(x).                #
    #                                                         #
    ###########################################################
    h = double(xEnd)/steps
    y[0,:] = y0

    for k in xrange(steps-1):
        F = lambda x: x - y[k] - h * rhs(x)
        y[k+1] = fsolve(F, y[k] + h * rhs(y[k]))
        x[k+1] = (k+1)*h

    if flag:
        return x[-1], y[-1][:]
    else:
        return x, y


def integrate_IM(y0, xStart, xEnd, steps, flag=False):
    r"""Integrate ODE with implicit midpoint rule

    Input: y0     ... initial condition
           xStart ... start x
           xEnd   ... end   x
           steps  ... number of steps (h = (xEnd - xStart)/N)
           flag   ... flag == False return complete solution: (phi, phi', t)
                      flag == True  return solution at endtime only: phi(tEnd)

    Output: x ... variable
            y ... solution
    """
    x = zeros(steps)
    y = zeros((steps, size(y0)))
    #################################################################
    #                                                               #
    # TODO: Implementieren Sie hier die implizite Mittelpunktsregel #
    #       zur integration der funktion y(x).                      #
    #                                                               #
    #################################################################
    h = double(xEnd)/steps
    y[0,:] = y0

    for k in xrange(steps-1):
        F = lambda x: x - y[k] - h * rhs(0.5*(x + y[k]))
        y[k+1,:] = fsolve(F, y[k] + h * rhs(y[k]))
        x[k+1] = (k+1)*h

    if flag:
        return x[-1], y[-1][:]
    else:
        return x, y


# Unteraufgabe c)

def integrate_VV(y0, xStart, xEnd, steps, flag=False):
    r"""Integrate ODE with velocity verlet rule

    Input: y0     ... initial condition
           xStart ... start x
           xEnd   ... end   x
           steps  ... number of steps (h = (xEnd - xStart)/N)
           flag   ... flag == False return complete solution: (phi, phi', t)
                      flag == True  return solution at endtime only: phi(tEnd)

    Output: x ... variable
            y ... solution
    """
    x = zeros(steps)
    y = zeros((steps, size(y0)))
    #############################################################
    #                                                           #
    # TODO: Implementieren Sie hier die velocity Verlet Methode #
    #       zur integration der funktion y(x).                  #
    #                                                           #
    #############################################################

    """ NOTE: This implementation expects the locations and velocities as
        its initial conditions. For this to work consistently, the initial
        conditions in ex d) for integrate_vv were altered accordingly. """

    D = 3
    q0, v0 = hsplit(y0, 2)
    v = zeros((steps, size(p0)))
    q = zeros((steps, size(q0)))
    h = double(xEnd)/steps

    v[0] = v0
    q[0] = q0

    for k in xrange(steps-1):
        q[k+1] = q[k] + h * v[k] + 0.5 * h**2 * rhs_vv(q[k])
        v[k+1] = v[k] + 0.5 * h * (rhs_vv(q[k]) + rhs_vv(q[k+1]))
        x[k+1] = (k+1)*h

    y = hstack((q, v))

    if flag:
        return x[-1], y[-1][:]
    else:
        return x, y


# Unteraufgabe d)

G = 1.0
m = array([500.0, 1.0])
q0 = hstack([0,0,0, 2,0,0])
p0 = hstack([0,0,0, 0,sqrt(m[0]/q0[3]),0])
y0 = hstack([q0, p0])

v0 = (p0.reshape((2,3)) / m[:,newaxis]).flatten() # divide by the masses
y0vv = hstack([q0, v0]) # for velocity vervlet

# Compute
T = 3
nrsteps = 5000

starttime = time()
t_ee, y_ee = integrate_EE(y0, 0, T, nrsteps, False)
endtime = time()
print('EE needed %f seconds' % (endtime-starttime))

starttime = time()
t_ie, y_ie = integrate_IE(y0, 0, T, nrsteps, False)
endtime = time()
print('IE needed %f seconds' % (endtime-starttime))

starttime = time()
t_im, y_im = integrate_IM(y0, 0, T, nrsteps, False)
endtime = time()
print('IM needed %f seconds' % (endtime-starttime))


starttime = time()
t_vv, y_vv = integrate_VV(y0vv, 0, T, nrsteps, False)
endtime = time()
print('VV needed %f seconds' % (endtime-starttime))

# Plot
fig = figure(figsize=(12,8))
ax = fig.gca()
ax.set_aspect("equal")

ax.plot(y_ee[:,0], y_ee[:,1], "b-")
ax.plot(y_ee[:,3], y_ee[:,4], "b-", label="EE")

ax.plot(y_ie[:,0], y_ie[:,1], "g-")
ax.plot(y_ie[:,3], y_ie[:,4], "g-", label="IE")

ax.plot(y_im[:,0], y_im[:,1], "r-")
ax.plot(y_im[:,3], y_im[:,4], "r-", label="IM")

ax.plot(y_vv[:,0], y_vv[:,1], "m-")
ax.plot(y_vv[:,3], y_vv[:,4], "m-", label="VV")

ax.grid(True)
ax.set_xlim(-3,3)
ax.set_ylim(-3,3)
ax.legend(loc="upper right")
ax.set_xlabel("x")
ax.set_ylabel("y")
fig.savefig("zwei.pdf")


# Unteraufgabe e)

G = 1.0
m = array([1.0, 1.0, 1.0])
q0 = hstack([0.97000436, -0.24308753, 0,
             -0.97000436, 0.24308753, 0,
             0, 0, 0])
p0 = hstack([0.93240737/2., 0.86473146/2., 0,
             0.93240737/2., 0.86473146/2., 0,
             -0.93240737, -0.86473146, 0])
y0 = hstack([q0, p0])

v0 = p0 # the masses are all 1, so nothing has to be done to get the velocity
y0vv = hstack([q0, v0])

t_vv, y_vv = integrate_VV(y0vv, 0, 2.0, 1000, False)

fig = figure(figsize=(12,8))
ax = fig.gca()
ax.set_aspect("equal")

ax.plot(y_vv[:,0], y_vv[:,1], "b-", label="m1")
ax.plot(y_vv[:,3], y_vv[:,4], "g-", label="m2")
ax.plot(y_vv[:,6], y_vv[:,7], "r-", label="m3")

ax.grid(True)
ax.set_xlim(-1.5,1.5)
ax.set_ylim(-1.5,1.5)
ax.legend(loc="upper right")
ax.set_xlabel("x")
ax.set_ylabel("y")
fig.savefig("drei.pdf")


# Unteraufgabe f)

G = 2.95912208286e-4

# Anfangswerte der Planeten
msun = 1.00000597682
qsun = array([0,0,0])
vsun = array([0,0,0])

mj = 0.00095486104043
qj = array([-3.5023653, -3.8169847, -1.5507963])
vj = array([0.00565429, -0.00412490, -0.00190589])

ms = 0.000285583733151
qs = array([9.0755314, -3.0458353, -1.6483708])
vs = array([0.00168318, 0.00483525, 0.00192462])

mu = 0.0000437273164546
qu = array([8.3101420, -16.2901086, -7.2521278])
vu = array([0.00354178, 0.00137102, 0.00055029])

mn = 0.0000517759138449
qn = array([11.4707666, -25.7294829, -10.8169456])
vn = array([0.00288930, 0.00114527, 0.00039677])

mp = 7.692307692307693e-09
qp = array([-15.5387357, -25.2225594, -3.1902382])
vp = array([0.00276725, -0.00170702, -0.00136504])

m = array([msun, mj, ms, mu, mn, mp])
q0 = hstack([qsun, qj, qs, qu, qn, qp])
p0 = hstack([msun*vsun, mj*vj, ms*vs, mu*vu, mn*vn, mp*vp])
y0 = hstack([q0, p0])

v0 = hstack([vsun, vj, vs, vu, vn, vp])
y0vv = hstack([q0, v0])

T = 20000
nrsteps = 2000


starttime = time()
t_ee, y_ee = integrate_EE(y0, 0, T, nrsteps, False)
endtime = time()
print('EE needed %f seconds for %f steps' % (endtime-starttime, nrsteps))

fig = figure(figsize=(12,8))
ax = fig.gca()
plot(y_ee[:,0], y_ee[:,1], "b-", label="Sonne")
plot(y_ee[:,3], y_ee[:,4], "g-", label="Jupiter")
plot(y_ee[:,6], y_ee[:,7], "r-", label="Saturn")
plot(y_ee[:,9], y_ee[:,10], "c-", label="Uranus")
plot(y_ee[:,12], y_ee[:,13], "m-", label="Neptun")
plot(y_ee[:,15], y_ee[:,16], "k-", label="Pluto")

plot(y_ee[-1,0], y_ee[-1,1], "bo")
plot(y_ee[-1,3], y_ee[-1,4], "go")
plot(y_ee[-1,6], y_ee[-1,7], "ro")
plot(y_ee[-1,9], y_ee[-1,10], "co")
plot(y_ee[-1,12], y_ee[-1,13], "mo")
plot(y_ee[-1,15], y_ee[-1,16], "ko")
grid(True)
xlim(-50, 50)
ylim(-50, 50)
ax.legend(loc="upper right")
ax.set_xlabel("x")
ax.set_ylabel("y")
savefig("solar_ee.pdf")


starttime = time()
t_ie, y_ie = integrate_IE(y0, 0, T, nrsteps, False)
endtime = time()
print('IE needed %f seconds for %f steps' % (endtime-starttime, nrsteps))

fig = figure(figsize=(12,8))
ax = fig.gca()
plot(y_ie[:,0], y_ie[:,1], "b-", label="Sonne")
plot(y_ie[:,3], y_ie[:,4], "g-", label="Jupiter")
plot(y_ie[:,6], y_ie[:,7], "r-", label="Saturn")
plot(y_ie[:,9], y_ie[:,10], "c-", label="Uranus")
plot(y_ie[:,12], y_ie[:,13], "m-", label="Neptun")
plot(y_ie[:,15], y_ie[:,16], "k-", label="Pluto")

plot(y_ie[-1,0], y_ie[-1,1], "bo")
plot(y_ie[-1,3], y_ie[-1,4], "go")
plot(y_ie[-1,6], y_ie[-1,7], "ro")
plot(y_ie[-1,9], y_ie[-1,10], "co")
plot(y_ie[-1,12], y_ie[-1,13], "mo")
plot(y_ie[-1,15], y_ie[-1,16], "ko")
grid(True)
xlim(-50, 50)
ylim(-50, 50)
ax.legend(loc="upper right")
ax.set_xlabel("x")
ax.set_ylabel("y")
savefig("solar_ie.pdf")


starttime = time()
t_im, y_im = integrate_IM(y0, 0, T, nrsteps, False)
endtime = time()
print('IM needed %f seconds for %f steps' % (endtime-starttime, nrsteps))

fig = figure(figsize=(12,8))
ax = fig.gca()
plot(y_im[:,0], y_im[:,1], "b-", label="Sonne")
plot(y_im[:,3], y_im[:,4], "g-", label="Jupiter")
plot(y_im[:,6], y_im[:,7], "r-", label="Saturn")
plot(y_im[:,9], y_im[:,10], "c-", label="Uranus")
plot(y_im[:,12], y_im[:,13], "m-", label="Neptun")
plot(y_im[:,15], y_im[:,16], "k-", label="Pluto")

plot(y_im[-1,0], y_im[-1,1], "bo")
plot(y_im[-1,3], y_im[-1,4], "go")
plot(y_im[-1,6], y_im[-1,7], "ro")
plot(y_im[-1,9], y_im[-1,10], "co")
plot(y_im[-1,12], y_im[-1,13], "mo")
plot(y_im[-1,15], y_im[-1,16], "ko")
grid(True)
xlim(-50, 50)
ylim(-50, 50)
ax.legend(loc="upper right")
ax.set_xlabel("x")
ax.set_ylabel("y")
savefig("solar_im.pdf")

starttime = time()
t_vv, y_vv = integrate_VV(y0vv, 0, T, nrsteps, False)
endtime = time()
print('VV needed %f seconds for %f steps' % (endtime-starttime, nrsteps))

fig = figure(figsize=(12,8))
ax = fig.gca()
plot(y_vv[:,0], y_vv[:,1], "b-", label="Sonne")
plot(y_vv[:,3], y_vv[:,4], "g-", label="Jupiter")
plot(y_vv[:,6], y_vv[:,7], "r-", label="Saturn")
plot(y_vv[:,9], y_vv[:,10], "c-", label="Uranus")
plot(y_vv[:,12], y_vv[:,13], "m-", label="Neptun")
plot(y_vv[:,15], y_vv[:,16], "k-", label="Pluto")

plot(y_vv[-1,0], y_vv[-1,1], "bo")
plot(y_vv[-1,3], y_vv[-1,4], "go")
plot(y_vv[-1,6], y_vv[-1,7], "ro")
plot(y_vv[-1,9], y_vv[-1,10], "co")
plot(y_vv[-1,12], y_vv[-1,13], "mo")
plot(y_vv[-1,15], y_vv[-1,16], "ko")
grid(True)
xlim(-50, 50)
ylim(-50, 50)
ax.legend(loc="upper right")
ax.set_xlabel("x")
ax.set_ylabel("y")
savefig("solar_vv.pdf")
