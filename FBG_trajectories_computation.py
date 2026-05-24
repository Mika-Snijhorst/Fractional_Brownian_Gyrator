import numpy as np
import matplotlib.pyplot as plt
import mpmath
from fbm import FBM
from scipy.special import gamma
from tqdm import tqdm
import os

#sets the precision of mpmath to 15 decimals  
mpmath.mp.dps = 15

#params
k = 1
eta_alpha = 1
eta_nu = 1
k_B = 1
Tf = 5
T0 = 1e-5
dt = 0.0005

T_x = 1
T_y_list = [1, 10, 100]
alpha_list = [1]
nu_list = [0.55, 0.65, 0.8, 0.9] 

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

def Hurst_and_amplitude(alpha, nu, T_x, T_y):
    '''
    Calculates the Hurst exponents and noise amplitudes for given parameters.
    '''
    Hurst_x = 1 - alpha/2
    Hurst_y = 1 - nu/2

    A_x = np.sqrt(2*k_B*T_x*eta_alpha/gamma(2*Hurst_x+1))
    A_y = np.sqrt(2*k_B*T_y*eta_alpha/gamma(2*Hurst_y+1))

    return Hurst_x, Hurst_y, A_x, A_y

def run_trajectory_computation(t_vals, Hurst_x, Hurst_y, A_x, A_y):
    #generates the FBM noise, t_vals[-1] gives the total time corresponding with T-1 steps
    T = len(t_vals)
    fbm_x = FBM(n=T-1, hurst = Hurst_x, length = t_vals[-1], method = "daviesharte")
    fbm_y = FBM(n=T-1, hurst = Hurst_y, length = t_vals[-1], method = "daviesharte")
    fbm_x_traj = fbm_x.fbm()
    fbm_y_traj = fbm_y.fbm()

    #the noise needs to be normalized for the right time steps
    xi_x = np.diff(fbm_x_traj) / dt
    xi_y = np.diff(fbm_y_traj) / dt

    #links Fractional Brownian noise to FDT, including a temperature dependence
    f_x = A_x*xi_x
    f_y = A_y*xi_y

    #the H functions governing the motion
    print("Inverting H_dx")
    H_dx = invert(H_dx_hat, t_vals)
    print("Inverting H_dy")
    H_dy = invert(H_dy_hat, t_vals)
    print("Inverting H_c")
    H_c  = invert(H_c_hat, t_vals)

    #this simplifies the convolution just like 1.10 from the thesis
    H_x = H_dx + H_c
    H_y = H_dy + H_c

    #this uses the convolution module included within numpy, we have to chip off the last value to match the noise array length
    pos_x = dt * (np.convolve(H_x, f_x, mode ="full")[:len(t_vals) - 1]
            + np.convolve(H_c, f_y, mode= "full")[:len(t_vals) - 1])

    pos_y = dt * (np.convolve(H_y, f_y, mode='full')[:len(t_vals)-1]
            + np.convolve(H_c, f_x, mode='full')[:len(t_vals)-1])
    return pos_x, pos_y

#create folder for trajectory data if it doesn't exist
os.makedirs("trajectory_data", exist_ok = True)

#loop through all possibilities so when plotting, all the data is already available
for alpha in alpha_list:
    for nu in nu_list:
        #only compute trajectories for alpha >= nu as we impose this condition
        if alpha >= nu:    
            for T_y in T_y_list:    
                '''
                the data will be stored in a folder named "trajectory_data", the data is named according to the following format:
                Data_for_trajectory_alpha{alpha}nu{nu}-Tx={T_x}Ty={T_y}-k={k}_Tf={Tf}_dt={dt}.npz
                '''
                #defines the path where the data will be stored
                folder = "trajectory_data"
                save = os.path.join(folder, f"Data_for_trajectory_alpha{alpha}nu{nu}-Tx={T_x}Ty={T_y}-k={k}_Tf={Tf}_dt={dt}.npz")

                #checks if the file already exists, if it does it skips the computation, otherwise it runs the computation and saves the data
                if os.path.exists(save):
                    print(f"file already exists, skipping computation for alpha = {alpha}, nu = {nu}, T_y = {T_y}")
                    continue

                #computes the trajectory
                Hurst_x, Hurst_y, A_x, A_y = Hurst_and_amplitude(alpha, nu, T_x, T_y)
                pos_x, pos_y = run_trajectory_computation(t_vals, Hurst_x, Hurst_y, A_x, A_y)
                
                #saves the data
                np.savez(save, pos_x = pos_x, pos_y = pos_y)
                print(f"computation complete, data saved to folder: {folder}")
                print(f"parameters: alpha = {alpha}, nu = {nu}, Tx = {T_x}, Ty = {T_y}")

print("--all computations complete--")