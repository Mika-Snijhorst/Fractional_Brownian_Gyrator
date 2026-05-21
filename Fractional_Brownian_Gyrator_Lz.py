#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 09:16:51 2026

@author: mika
"""

import numpy as np
import matplotlib.pyplot as plt
import mpmath
from scipy.integrate import cumulative_trapezoid

#sets the precion of mpmath to 15 decimals
mpmath.mp.dps = 15

#params
k = 1
eta_alpha = 1
eta_nu = 1
k_B = 1

#variables
alpha = 1.0
nu = 1.0
T_x = 1
T_y = 2

#time array
t_vals = np.linspace(1e-5, 10.0, 200)

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

#Laplace transforms of the convolutions
def Xi_3_hat(s):        return 2 * k * eta_alpha * s**(alpha - 1) * G_hat(s)
def Xi_4_hat(s):        return eta_alpha * eta_nu * s**(alpha + nu - 1) * G_hat(s)
def Delta_4_hat(s):     return 2 * k * eta_nu * s**(nu - 1) * G_hat(s)

#Laplace transforms of the derivatives
def Xi_4_dot_hat(s):    return s * Xi_4_hat(s)  
def H_c_dot_hat(s):     return s * H_c_hat(s)

def invert(func_hat, t_values):
    '''
    Inverse Laplace transform function, it uses the Talbot method to 
    evaluate the input function at every time value that is within the input array.
    Because the Talbot method requires t > 0, I built a little logic gate (tho not 
    explicitily necessary for our calculation). mpmath uses its own floats, so we 
    convert the value back to a python float first so the dtype of t_vals matches the
    d_type of the output functions
    '''
    array = np.zeros_like(t_values)     #sets up an array to avoid copying
    for i, t in enumerate(t_values):
        if t<= 0:
            raise ValueError("Talbot requires t > 0")
        else:
            array[i] = float(mpmath.invertlaplace(func_hat, t, method = "talbot"))
    return array

#%%

def run(t):
    '''
    Takes time values as an input and then evaluates the functions in the Laplace domain 
    at those time values. The integration is done by the cumulative_trapezoid() function
    from Scipy. After the functions are evaluated we run the integral 

    '''
    #inverse Laplace transform of the H-functions, dissipation of x, y and the coupling respectively
    H_dx = invert(H_dx_hat, t)
    H_dy = invert(H_dy_hat, t)
    H_c = invert(H_c_hat, t)
    
    #inverse Laplace transforms of the convolutions
    Xi_3 = invert(Xi_3_hat, t)
    Xi_4 = invert(Xi_4_hat, t)
    Delta_4 = invert(Delta_4_hat, t)
    
    #inverse Laplace transform of the derivatives
    H_c_dot = invert(H_c_dot_hat, t)
    Xi_4_dot = invert(Xi_4_dot_hat, t)
    
    #the first "initial" term
    term_1 = k_B * T_x * H_dx * Xi_3 - k_B * T_y * H_dy * Delta_4
    
    #the second "dissipative" term
    integrand_2 = H_dx * H_dy
    integral_2 = cumulative_trapezoid(integrand_2, t, initial = 0)
    term_2 = -4 * k * k_B * (T_x - T_y) * integral_2
    
    #the third "spring" term
    integrand_3 = H_c * Xi_4_dot - H_c_dot * Xi_4
    integral_3 = cumulative_trapezoid(integrand_3, t, initial = 0)
    term_3 = k_B * (T_x - T_y) * integral_3
    
    #L_z
    L_z = term_1 + term_2 + term_3
    
    return L_z, term_1, term_2, term_3

#calls the evaluation to run
L_z, term_1, term_2, term_3 = run(t_vals)

#plotting
plt.plot(t_vals, term_1, label = "Term 1")
plt.plot(t_vals, term_2, label = "Term 2")
plt.plot(t_vals, term_3, label = "Term 3")
plt.plot(t_vals, L_z, label = r"Total $\langle L_z\rangle$", linestyle = "--", color = "red")

#plot settings
plt.xlabel(r"Time $\dfrac{t}{\tau_c}$")
plt.ylabel(r"$\langle L_z\rangle$")
plt.legend()
plt.grid(True, linestyle = "--")
plt.axhline(0, color = "black", linewidth = 0.5)
plt.show()