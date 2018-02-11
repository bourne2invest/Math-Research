# -*- coding: utf-8 -*-
## This program is to implement a Backward Euler finite-difference approximation
## to the (forward-parabolic) Black-Scholes PDE:
##      V_t = 0.5 * sigma^2 * s^2 V_{ss} + r * ( s * V_s - V )
## subject to the boundary condition:
##      V(s,T) = f(s), for a smooth function of the stock price, s.
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from math import log

# Parameters for uniform grid
T = 1 #int(raw_input('Enter an integer value for the terminal time of the contract, T: ')) # terminal time ;; for e.g: if s_end = 2*k, let T = 1,
N = 100 #int(raw_input('Enter an integer value for the number of spatial nodes, N: ')) # spatial values ;; let N = 100,
M = 10000 #int(raw_input('Enter an integer value for the number of time steps, M: ')) # time values/hops (M ~ N^2) ;;  and let M = 10,000.

r = 0 # riskless interest rate
sigma = 0.18 # annual stock volatility
k = 25 # strike price in dollars

# uniform mesh in s
s_init = 0.5
s_end = 2*k #int(raw_input('Enter a value for s_end that is an integer-multiple of the strike price, n*k s.t: n is greater than 1: ')) * k
h = float(s_end - s_init) / N # step-size

# arrange grid in s, with N+1 equally spaced nodes
s = np.zeros(N+1)
s = np.linspace(s_init, s_end, num=N+1, endpoint=True)
s[0] = s_init
#print s[100] -->> s[100] == 50 (good)

# time discretization
t_init = 0
t_end = T
dt = float(t_end - t_init) / M

t = np.arange(t_init, t_end, dt) #arrange grid in time with step size dt
t[0] = t_init


# Define Diffusion and Drift coefficients/constants
D = 1 # In Black-Scholes we have:
      # D = ( 0.5 * (sigma**2) * (s**2) ),
C = 0 # C = ( r * s )

# Define ratios for matrix algorithm
Lambda = ( (D*dt) / (h**2) )

mu = ( (C*dt) / (2*h) )
#print mu, Lambda

# define the tridiagonal matrix A
A = np.zeros((N+1, N+1), dtype=float)
A[0,0] = (1 + 2*Lambda) # need to eventually implement this form: (1 + 2*Lambda + dt * r**(m+1)), since we have r = 0
A[0,1] = (- Lambda - mu) 
for n,l in enumerate(((mu - Lambda), (1 + 2*Lambda), (- Lambda - mu) )):
    np.fill_diagonal(A[1:,n:], l)
A[N,N-1] = (mu - Lambda)
A[N,N] = (1 + 2*Lambda)
#print(A)

        
# definition of the solution u(s,t) to our PDE
u = np.zeros((N+1, M+1)) # NxM array to store values of the solution ;; u = np.zeros((N+1, M+1))

# Initial Condition
for j in range(0, N+1):
    if (s[j] - k) <= 0: # s[j] - k, for a Call-Option // OR:  (k - s[j]) for a PUT-Option
       u[j,0] = 0
    elif (s[j] - k) > 0: 
        u[j,0] = s[j] - k
    #print u[:,0] # print the initial condition
#print( (s[0] - k, s[45] - k, s[55] - k, s[100] - k) )

#plot the Initial Condition
    #plt.plot(s, u[:,0], label = 'Initial Condition')
    #plt.legend()
    #plt.show()

# Backward Euler/Inversion scheme                                                     
for m in range(0, M):
    u[:,m+1] = np.linalg.solve(A, u[:,m])   # numpy.linalg.solve(a, b) :: Solves a linear matrix equation, or system of linear scalar equations.
    u[0,m+1] = 0                            # Computes the “exact” solution, x, of the well-determined, i.e., full rank, linear matrix equation ax = b.
    u[N,m+1] = (s_end - k)                  # u[N,m+1] = (s_end - k) yields a better approximation than u[0,m+1] = 0.

            
# define explicit Black-Scholes function
BS = np.zeros((N+1, M+1)) #array has same size as the solution u
x_1 = np.zeros((N+1, M+1))
x_2 = np.zeros((N+1, M+1))
for m in range(1, M+1):
    for j in range(0, N+1):
        x_1[j,m] = (log(s[j] / k) + (r + 0.5 * pow(sigma,2.0)) * (m * dt))  / (sigma * np.sqrt(m * dt))
        x_2[j,m] = (log(s[j] / k) + (r - 0.5 * pow(sigma,2.0)) * (m * dt)) / (sigma * np.sqrt(m * dt))
        BS[j,m] = s[j] * norm.cdf(x_1[j,m], loc=0, scale=1) - k * np.exp(-r * (m*dt)) * norm.cdf(x_2[j,m], loc=0, scale=1)

#print x_1, x_2

# Compute the Error
#E = np.zeros((N+1, M+1)) # so that E has NxM entries--one for each time-step and node in space
#for j in range(33, 70): #for j in range(38,62) to min. error; 
#    E_new = np.absolute(BS[j,m] - u[j,m])
#    if (np.any(E_new >= E)): 
#        E = E_new
#    else:
#        E = E
#    print(E)
       
print( 'Sup-norm Error = ', np.max(np.abs(BS - u)) )    # computes the sup-norm or max-norm of the arrays
#print 'Error = ' np.linalg.norm(BS - u, ord=np.infty) #// alternative way to compute the sup-norm

# 2D plot
#print s.shape, u.shape, u[:,0].shape, u[:,M].shape
plt.plot(s, u, label = 'approximate solution') # numerical approx. to solution of B-S PDE plotted versus s (stock price in dollars) ;; to plot Initial Condition use: plt.plot(s, u[:,0])
#plt.plot(s, BS, label = 'exact solution') #to plot for FIXED times use: plt.plot(s, BS[:,M/2], label = 'exact solution')
plt.title('Backward Euler B-S approx. with Truncation at Boundary') #plt.title('Backward Euler approx. to Black-Scholes with truncation at boundary')
plt.ylabel('Option Value (in dollars)')
plt.xlabel('Stock price, s (in dollars)')
#plt.legend()
plt.show()