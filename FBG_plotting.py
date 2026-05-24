import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

#params
k = 1
eta_alpha = 1
eta_nu = 1
k_B = 1
Tf = 5
T0 = 1e-5
dt = 0.0005

#variables
alpha = 1
nu = 1

#temperatures
T_x = 1
T_y_list = [1, 10, 100]

def load_trajectory_data(alpha, nu, T_x, T_y, k, Tf, dt):
    '''
    Loads the trajectory data for given parameters. The data is named according to the following format:
    Data_for_trajectory_alpha{alpha}nu{nu}-Tx={T_x}Ty={T_y}-k={k}_Tf={Tf}_dt={dt}.npz
    From this we can pull the correct dataset for ease of plotting.
    '''
    folder_trajectory = "trajectory_data"
    filename_trajectory = f"Data_for_trajectory_alpha{alpha}nu{nu}-Tx={T_x}Ty={T_y}-k={k}_Tf={Tf}_dt={dt}.npz"

    #loads the data and stores it
    data = np.load(os.path.join(folder_trajectory, filename_trajectory))
    pos_x = data['pos_x']
    pos_y = data['pos_y']
    return pos_x, pos_y

def load_angular_momentum_data(alpha, nu, T_x, T_y, k, Tf, dt):
    '''
    Loads the angular momentum data for given parameters. The data is named according to the following format:
    Angular_momentum_for_alpha{alpha}nu{nu}-Tx={T_x}Ty={T_y}-k={k}_Tf={Tf}_dt={dt}.npz
    From this we can pull the correct dataset for ease of plotting.
    '''
    folder_angular_momentum = "angular_momentum_data"
    filename_angular_momentum = f"Angular_momentum_for_alpha{alpha}nu{nu}-Tx={T_x}Ty={T_y}-k={k}_Tf={Tf}_dt={dt}.npz"

    #loads the data and stores it
    data = np.load(os.path.join(folder_angular_momentum, filename_angular_momentum))
    L_z = data['L_z']
    return L_z

#creates a subplot for eacht temperature, ax.flatten() allows to iterate without having to swap x and y
fig, ax = plt.subplots(1, len(T_y_list), figsize= (8*len(T_y_list), 3), sharex =True,sharey=True)
ax_flat = ax.flatten()

for i, T_y in enumerate(T_y_list):
    #load data for trajectories
    pos_x, pos_y = load_trajectory_data(alpha, nu, T_x, T_y, k, Tf, dt)

    #generates the colors for the colorbar, they are generated with respect to time
    colors = np.linspace(0, Tf, len(pos_x))

    #plots the trajectory
    sc = ax_flat[i].scatter(pos_x, pos_y, c=colors, cmap="plasma_r", s=10, alpha=0.7)
    ax_flat[i].plot(pos_x, pos_y, color = "black", linewidth = 0.5, alpha = 0.5)
    ax_flat[i].set_xlabel(r"$x(t)$")
    ax_flat[i].set_title(fr"$\alpha = {alpha}, \nu = {nu}$, T$_x$ = {T_x}, T$_y$ = {T_y}") 
    ax_flat[i].axvline(0, color = "black", linestyle = "--", linewidth = 0.5)
    ax_flat[i].axhline(0, color = "black", linestyle = "--", linewidth = 0.5)
    ax_flat[i].set_xlim(min(pos_x)-3, max(pos_x)+3)
    ax_flat[i].set_ylim(min(pos_y)-3, max(pos_y)+3)
    
    
    #plots the angular momentum in an inset
    L_z = load_angular_momentum_data(alpha, nu, T_x, T_y, k, Tf, dt)
    inset_ax=inset_axes(ax_flat[i], width="30%", height="30%", loc="upper left", borderpad = 1)
    inset_ax.plot(np.linspace(T0, Tf, len(L_z)), L_z, color = "blue", linewidth = 1)
    inset_ax.axhline(0, color = "black", linestyle = "--", linewidth = 0.5)
    inset_ax.set_xticks([])
    inset_ax.tick_params(axis = "y",labelsize=5, direction="in", pad=-15)

cbar = fig.colorbar(sc, ax=ax.ravel().tolist(), orientation="vertical", fraction=0.02, pad=0.02)
cbar.set_label("Time", rotation=270, labelpad=15)
plt.show()


