import numpy as np
import matplotlib.pyplot as plt
import mpmath
from fbm import FBM
from scipy.special import gamma
from tqdm import tqdm

#sets the precision of mpmath to 15 decimals  
mpmath.mp.dps = 15

#params
k = 0
eta_alpha = 1
eta_nu = 1
k_B = 1
Tf = 5
T0 = 1e-5
dt = 0.0005

#variables
alpha = 0.9
nu = 0.9

Hurst_x = 1 - alpha/2
Hurst_y = 1 - nu/2

T_x = 1
T_y = 100

A_x = np.sqrt(2*k_B*T_x*eta_alpha/gamma(2*Hurst_x+1))
A_y = np.sqrt(2*k_B*T_y*eta_alpha/gamma(2*Hurst_y+1))

#time array, we start from a small value because as t approaches 0, s approaches infinity.
t_vals = np.arange(T0, Tf, dt)

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
    for i, t in tqdm(enumerate(t_values)):
        if t<= 0:
            raise ValueError("Talbot requires t > 0")
        else:
            array[i] = float(mpmath.invertlaplace(func_hat, t, method = "talbot"))
    return array

#finds the dt
T = len(t_vals)


#the fbm package uses the Hurst exponent, so we need to convert alpha and nu

#generates the FBM noise, t_vals[-1] gives the total time corresponding with T-1 steps
fbm_x = FBM(n=T-1, hurst = Hurst_x, length = t_vals[-1], method = "daviesharte")
fbm_y = FBM(n=T-1, hurst = Hurst_y, length = t_vals[-1], method = "daviesharte")
fbm_x_traj = fbm_x.fbm()
fbm_y_traj = fbm_y.fbm()

#the noise needs to be normalized for the right time steps
xi_x = np.diff(fbm_x_traj) / dt
xi_y = np.diff(fbm_y_traj) / dt

f_x = A_x*xi_x
f_y = A_y*xi_y

#the H functions from before
print("Inverting H_dx")
H_dx = invert(H_dx_hat, t_vals)
print("Inverting H_dy")
H_dy = invert(H_dy_hat, t_vals)
print("Inverting H_c")
H_c  = invert(H_c_hat, t_vals)

#this simplifies the convolution just like 1.10
H_x = H_dx + H_c
H_y = H_dy + H_c

#this uses the convolution module included within numpy, we have to chip off the last value to match the noise array length
pos_x = dt * (np.convolve(H_x, f_x, mode ="full")[:len(t_vals) - 1]
          + np.convolve(H_c, f_y, mode= "full")[:len(t_vals) - 1])

pos_y = dt * (np.convolve(H_y, f_y, mode='full')[:len(t_vals)-1]
          + np.convolve(H_c, f_x, mode='full')[:len(t_vals)-1])

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
plt.savefig(f"trajectory_alpha{alpha}nu{nu}-Tx{T_x}Ty{T_y}-k{k}_Tf{Tf}_dt{dt}.png")
plt.show()












