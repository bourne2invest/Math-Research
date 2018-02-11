## This program is to implement a Finite Difference Approxiamtion to the 
## (forward-parabolic) Black-Scholes PDE:
##      V_t = 0.5 * sigma^2 * s^2 V_{ss} + r * ( s * V_s - V )
## subject to the boundary condition:
##      V(s,T) = f(s), for a smooth function of the stock price, s.
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from math import log

# Parameters for uniform grid
T = int(raw_input('Enter an integer value for the terminal time of the contract, T: ')) # terminal time ;; for e.g: if s_end = 2*k, let T = 1,
N = int(raw_input('Enter an integer value for the number of spatial nodes, N: ')) # spatial values ;; let N = 100,
M = int(raw_input('Enter an integer value for the number of time steps, M: ')) # time values/hops (M ~ N^2) ;;  and let M = 10,000.

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
# print s[100]

# time discretization
t_init = 0
t_end = T
dt = float(t_end - t_init) / M

# Check that Stability condition is satisfied:
if ( dt <= ((h**2) / ( (sigma**2) * (4 * (k**2)) )) ):    # Stability condition: dt <= (h**2) / ( (sigma**2) * (4 * (k**2)) )
   t = np.arange(t_init, t_end, dt)    #arrange grid in time with step size dt
   t[0] = t_init 
else:
    M = int(raw_input('Stability condition not satisfied. The chosen value for M returns a numerical approximation that does not converge. Enter a larger value for the number of time-steps, M: '))

# definition of Black-Scholes coefficients, alpha & beta
alpha = np.zeros(N+1)
gamma = np.zeros(N+1)
beta = float(2.0/3) # C.E.V. coefficient; NOTE: beta == 1 corresponds to Black-Scholes model; when beta < 1, use something to the effect of beta = float(1.0/2), or float(2.0/3)
for j in range(1, N):
    alpha[j] = ( (sigma) * ((s[j])**beta) )    
    gamma[j] = r * s[j]
    
# definition of the solution u(s,t) to our PDE
u = np.zeros((N+1, M+1)) # NxM array to store values of the solution ;; u = np.zeros((N+1, M+1))

# Initial Condition
for j in range(0, N+1):
    if (s[j] - k) <= 0: # s[j] - k, for a Call-Option // OR:  (k - s[j]) for a PUT-Option
       u[j,0] = 0
    elif (s[j] - k) > 0: 
        u[j,0] = s[j] - k
    print u[:,0] # print the initial condition
# print s[1] - k

# Finite Difference Scheme                                                       
for m in range(0, M):
    for j in range(0, N):
        if j == 0:
            u[j,m] = 0 # Boundary condition; setting equal to k yields a better approximation for the PUT-Option
        elif j == N:
            u[j+1,m] = 0 # Boundary Condition; force the value of the approximation to be the intial condition at s_end <==> j == N+1
        else:
            u[j,m+1] = u[j,m] + (( ((alpha[j])**2) * dt)/(2 * h**2)) * ( u[j+1,m] - 
            2 * u[j,m] + u[j-1,m] ) + ((gamma[j]) * ((dt)/(2*h))) * ( u[j+1,m] - u[j-1,m])
            - (r * dt) * u[j,m]
            
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
E = np.zeros((N+1, M+1)) # so that E has NxM entries--one for each time-step and node in space
for j in range(33, 70): #for j in range(38,62) to min. error; 
    E_new = np.absolute(BS[j,m] - u[j,m])
    if (np.any(E_new >= E)): 
        E = E_new
    else:
        E = E
    print E
       
print 'Error = ', np.max(np.abs(BS - u)) # computes the sup-norm or max-norm of the arrays
#print 'Error = ', np.linalg.norm(BS - u, ord=np.infty) // alternative way to compute the sup-norm

# 2D plot
plt.plot(s, u) # numerical approx. to solution of B-S PDE plotted versus s (stock price in dollars)
#plt.plot(s, BS, label = 'exact solution') #to plot for FIXED times use: plt.plot(s, BS[:,M/2], label = 'exact solution')
plt.title('CEV approx. with beta = 1 and truncation at boundary')
plt.ylabel('Option Value (in dollars)')
plt.xlabel('Stock price, s (in dollars)')
#plt.legend()
plt.show()