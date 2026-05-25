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
fig, ax = plt.subplots(1, len(T_y_list), figsize= (8*len(T_y_list), 4), sharex =True,sharey=True)
ax_flat = ax.flatten()
labels = ["a)", "b)", "c)", "d)", "e)"]

for i, T_y in enumerate(T_y_list):
    #load data for trajectories
    pos_x, pos_y = load_trajectory_data(alpha, nu, T_x, T_y, k, Tf, dt)

    #generates the colors for the colorbar, they are generated with respect to time
    colors = np.linspace(0, Tf, len(pos_x))
    cmap = "plasma_r"

    #plots the trajectory
    sc = ax_flat[i].scatter(pos_x, pos_y, c=colors, cmap=cmap, s=10, alpha=0.7)
    ax_flat[i].plot(pos_x, pos_y, color = "black", linewidth = 0.5, alpha = 0.5)
    ax_flat[i].set_xlabel(r"$x(t)$")
    ax_flat[i].axvline(0, color = "black", linestyle = "--", linewidth = 0.5)
    ax_flat[i].axhline(0, color = "black", linestyle = "--", linewidth = 0.5)
    ax_flat[i].set_xlim(min(pos_x)-3, max(pos_x)+3)
    ax_flat[i].set_ylim(min(pos_y)-3, max(pos_y)+3)
    print(max(pos_x)+3, min(pos_y)-3)

    #text in plot
    ax_flat[i].text(-9.8, -18, labels[i], fontsize = 10, color = "black")
    ax_flat[i].text(9.2, -16.5, f"T$_y$ = {T_y}", fontsize = 10, color = "black", ha = "right", va = "bottom")
    ax_flat[i].text(9.2, -18.5, fr"$\alpha$ = {alpha}, $\nu$ = {nu}", fontsize = 10, color = "black", ha = "right", va = "bottom")
    ax_flat[i].text(9.2, -13.5, f"k = {k}", fontsize = 10, color = "black", ha = "right", va = "bottom")
    ax_flat[i].tick_params(axis = "both", labelsize = 8, direction = "out")
    
    
    #plots the angular momentum in an inset
    L_z = load_angular_momentum_data(alpha, nu, T_x, T_y, k, Tf, dt)
    inset_ax=inset_axes(ax_flat[i], width="30%", height="30%", loc="upper left", borderpad = 2)
    inset_ax.scatter(np.linspace(T0, Tf, len(L_z)), L_z, c=np.linspace(0, Tf, len(L_z)), cmap=cmap, s=1, alpha=0.7)
    inset_ax.plot(np.linspace(T0, Tf, len(L_z)), L_z, color = "black", linewidth = 1, alpha = 0.5)
    inset_ax.axhline(0, color = "black", linestyle = ":", linewidth = 0.5)
    inset_ax.set_xticks([])
    inset_ax.tick_params(axis = "y",labelsize=5.5, direction="in")
    if L_z[(len(L_z)-5)] == 0:
        inset_ax.text(5, -0.05, r"$\langle L_z(t)\rangle$", ha="right", va="bottom", fontsize = 10, color = "black")
    else:    
        inset_ax.text(5, 0, r"$\langle L_z(t)\rangle$", ha="right", va="bottom", fontsize = 10, color = "black")

#sets the colorbar to the right of the last subplot, with a label and ticks
cbar = fig.colorbar(sc, ax=ax.ravel().tolist(), orientation="vertical", fraction=0.02, pad=0.02)
cbar.set_label("Time", rotation=270, labelpad=15)
ax_flat[0].set_ylabel(r"$y(t)$")

folder = "Thesis_plots"
filename = f"plot_alpha{alpha}nu{nu}-Tx={T_x}-k={k}_Tf={Tf}_dt={dt}.png"
plt.savefig(os.path.join(folder, filename))


plt.show()


