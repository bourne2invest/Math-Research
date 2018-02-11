## This program is to implement a Finite Difference method approximation
## to solve the Heat Equation, u_t = k * u_xx,
## in 1D w/out sources & on a finite interval 0 < x < L. The PDE
## is subject to B.C: u(0,t) = u(L,t) = 0,
## and the I.C: u(x,0) = f(x).
import numpy as np
import matplotlib.pyplot as plt
from math import sin
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Parameters    
L = 1 # length of the rod
T = 10 # terminal time
N = 100 # spatial values
M = 10000 # time values/hops; (M ~ N^2)
s = 0.25 # s := k * ( (dt) / (dx)^2 )

# uniform mesh
x_init = 0
x_end = L
dx = float(x_end - x_init) / N

x = np.arange(x_init, x_end, dx)
x[0] = x_init

# time discretization
t_init = 0
t_end = T
dt = float(t_end - t_init) / M

t = np.arange(t_init, t_end, dt)
t[0] = t_init

# time-vector
for m in xrange(0, M):
    t[m] = m * dt
 
# spatial-vector
for j in xrange(0, N):
    x[j] = j * dx

# definition of the solution u(x,t) to u_t = k * u_xx
u = np.zeros((N, M+1)) # array to store values of the solution

# Finite Difference Scheme:

u[:,0] = x * (x - 1) #initial condition
              
for m in xrange(0, M):
    for j in xrange(1, N-1):
        if j == 1:
            u[j-1,m] = 0 # Boundary condition
        elif j == N-1:
            u[j+1,m] = 0 # Boundary Condition
        else:
            u[j,m+1] = u[j,m] + s * ( u[j+1,m] - 
            2 * u[j,m] + u[j-1,m] )

# for 2D plot    
#print u
plt.plot(x, u)
plt.title('Finite Difference Approx. to Heat Equation with 40 mesh points')
plt.xlabel('Length of rod, x')
plt.ylabel('The solution, u(x,t)')
plt.show()

#color graph
#corrected_u = u[:,:-1:]
#plt.pcolor(t, x, corrected_u)
#plt.title('Finite Difference Approx. to Heat Equation with 40 mesh points')
#plt.xlabel('Time, t (seconds)')
#plt.ylabel('Length of rod, x (meters)')
#plt.show()

#correction to array u
t, x = np.meshgrid(t, x)
u = u[:,:-1]

# for 3D graph
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x, t, u, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.title('Numerical solution to Heat Equation with 40 mesh points')
plt.ylabel('Time, t (seconds)')
plt.xlabel('Length of rod, x (meters)')
plt.show()