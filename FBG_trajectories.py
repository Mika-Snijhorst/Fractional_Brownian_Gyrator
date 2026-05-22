import numpy as np
import matplotlib.pyplot as plt
import mpmath
from fbm import FBM

#sets the precision of mpmath to 15 decimals  
mpmath.mp.dps = 15

#params
k = 1
eta_alpha = 1
eta_nu = 1
k_B = 1

#variables
alpha = 0.8
nu = 0.7
T_x = 1
T_y = 3

#time array, we start from a small value because as t approaches 0, s approaches infinity.
t_vals = np.linspace(1e-5, 0.5, 150)

def G_hat(s):
    '''
    The main resolvent function
    '''
    return 1 / (eta_alpha * eta_nu * s**(alpha+nu) 
                + 2 * k * eta_alpha * s**alpha + 2 * k * eta_nu * s**nu)

#Laplace transforms of the H-functions, dissipation of x, dissipation of y and coupling respectively
def H_dx_hat(s):        return eta_nu * s**nu * G_hat(s)
def H_dy_hat(s):        return eta_alpha * s**alpha * G_hat(s)
def H_c_hat(s):         return 2 * k * G_hat(s)

def invert(func_hat, t_values):
    '''
    Inverse Laplace transform function, it uses the Talbot method to 
    evaluate the input function at every time value that is within the input array.
    Because the Talbot method requires t > 0, I built a little logic gate (tho not 
    explicitily necessary for our calculation). mpmath uses its own floats, so we 
    convert the value back to a python float first so the dtype of t_vals matches the
    d_type of the output functions
    '''
    array = np.zeros_like(t_values)     
    for i, t in enumerate(t_values):
        if t<= 0:
            raise ValueError("Talbot requires t > 0")
        else:
            array[i] = float(mpmath.invertlaplace(func_hat, t, method = "talbot"))
    return array

#finds the dt
T = len(t_vals)
dt = t_vals[1] - t_vals[0]

#the fbm package uses the Hurst exponent, so we need to convert alpha and nu
Hurst_x = 1 - alpha/2
Hurst_y = 1 - nu/2

#generates the FBM noise, t_vals[-1] gives the total time corresponding with T-1 steps
fbm_x = FBM(n=T-1, hurst = Hurst_x, length = t_vals[-1], method = "daviesharte")
fbm_y = FBM(n=T-1, hurst = Hurst_y, length = t_vals[-1], method = "daviesharte")
f_x = fbm_x.fbm()
f_y = fbm_y.fbm()

#the noise needs to be normalized for the right time steps
xi_x = np.diff(f_x) / dt
xi_y = np.diff(f_y) / dt

#the H functions from before
H_dx = invert(H_dx_hat, t_vals)
H_dy = invert(H_dy_hat, t_vals)
H_c  = invert(H_c_hat, t_vals)

#this simplifies the convolution just like 1.10
H_x = H_dx + H_c
H_y = H_dy + H_c

#this uses the convolution module included within numpy, we have to chip off the last value to match the noise array length
pos_x = dt * (np.convolve(H_x, xi_x, mode ="full")[:len(t_vals) - 1]
          + np.convolve(H_c, xi_y, mode= "full")[:len(t_vals) - 1])

pos_y = dt * (np.convolve(H_y, xi_y, mode='full')[:len(t_vals)-1]
          + np.convolve(H_c, xi_x, mode='full')[:len(t_vals)-1])

#a bit crude but it makes it more intuitive to see the trajectory evolve in time
colors = ["red", "orange", "yellow", "green", "blue", "purple", "magenta"]
ph = len(pos_x) // len(colors)
for i in range(len(colors)):
    plt.scatter(pos_x[i*ph], pos_y[i*ph], color=colors[i], s = 6)
    plt.plot(pos_x[i*ph:(i+1)*ph+1], pos_y[i*ph:(i+1)*ph+1], color=colors[i], lw = "0.5")

#plotting settings
plt.xlabel(r"$x(t)$")
plt.ylabel(r"$y(t)$")
plt.axvline(0, color = "black", lw = "0.5", alpha = 0.7)
plt.axhline(0, color = "black", lw = "0.5", alpha = 0.7)
plt.show()












