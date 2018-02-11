# -*- coding: utf-8 -*-
## This program is to implement a Backward (Implicit) Euler Method to
## approximate the solution of the Heat/Diffusion Equation, u_t = k * u_xx,
## in 1D without sources & on a finite interval 0 < x < L. The PDE is subject
## to the B.C: u(0,t) = u(L,t) = 0,
## and the I.C: u(x,0) = f(x).

import numpy as np
import matplotlib.pyplot as plt
from math import pi
#from numpy.linalg import inv
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Parameters    
L = 1 # length of the rod
T = 10 # terminal time
N = 100 # spatial values
M = 10000 # time values/hops; (M ~ N^2)
s = 0.25 # s := k * ( (dt) / (dx)^2 )
kappa = 1 # Diffusivity constant

# uniform mesh in space
x_init = 0
x_end = L
dx = float(x_end - x_init) / (N+1)

x = np.arange(x_init, x_end, dx)
x[0] = x_init

# time discretization
t_init = 0
t_end = T
dt = float(t_end - t_init) / (M+1)

t = np.arange(t_init, t_end, dt)
t[0] = t_init


# define the tridiagonal matrix A
A = np.zeros((N+1, N+1), dtype=float)
A[0,0] = (1 + 2*s)
A[0,1] = -s
for k,l in enumerate((-s, (1 + 2*s), -s)):
    np.fill_diagonal(A[1:,k:], l)
A[N,N-1] = -s
A[N,N] = (1 + 2*s)
#print A


# definition of the solution u(x,t) to u_t = k * u_xx
u = np.zeros((N+1, M+2)) # array to store values of the APPROXIMATE solution

u[:M+1,0] = np.sin(2*pi*x) #initial condition; let the I.C. be constant for now...; for periodic B.C. use np.sin(x) or np.cos(x)
#print u[:,0]


# Inversion/Backward Euler scheme
for m in range(0, M+1):
    u[:,m+1] = np.linalg.solve(A, u[:,m])   # numpy.linalg.solve(a, b) :: Solves a linear matrix equation, or system of linear scalar equations.
    u[0,m+1] = 0                            # Computes the “exact” solution, x, of the well-determined, i.e., full rank, linear matrix equation ax = b.
    u[N,m+1] = 0                            # Boundary Conditions (line 60 - 61)
#print u[N,:]

#inv_A = np.linalg.matrix_power(np.matrix(A), -M)  
#print inv_A

#correction to array u
t, x = np.meshgrid(t, x)
u = u[:,:-1]

#print 'Error = ', np.max(np.abs(BS - u)) # computes the sup-norm or max-norm of the arrays
#print 'Error = ', np.linalg.norm(BS - u, ord=np.infty) // alternative way to compute the sup-norm

# Compute the Error
u_exact = np.zeros((N+1, M+2)) # array to store values of the EXACT solution
u_exact = np.sin(2*pi*x) * np.exp(-4* pow(pi,2.0) * kappa * t)

# Print the Error
print 'Error under Infinity-norm = ', np.max(np.abs(u_exact - u)) # computes the sup-norm or max-norm of the arrays
print 'Error under 2-norm = ', np.linalg.norm(u_exact - u, ord=2) #np.linalg.norm(BS - u, ord=np.infty) // alternative way to compute the sup-norm, or the 2-norm

# for 2D plot    
print x.shape, u.shape # prints shape/dimension of an array
plt.plot(x, u)
#plt.plot(x, u[:,M/20], label = 'Approximate Solution') #to plot for FIXED times use: plt.plot(s, BS[:,M/2], label = 'exact solution')
plt.title('Backward Euler for Heat Equation at t = T/20')
plt.xlabel('Length of rod, x')
plt.ylabel('The temperature, u(x,t)')
#plt.legend()
plt.show()

# for 3D graph
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#surf = ax.plot_surface(x, t, u, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.5, aspect=5)
#plt.title('Numerical solution to Heat Equation with 100 mesh points')
#plt.ylabel('Time, t (seconds)')
#plt.xlabel('Length of rod, x (meters)')
#plt.show()