import numpy as np

def initialisation(n_dim, xdim, ydim, xcells, ycells, P0, adiab_ind):
    """For a given grid, initialise the simulations"""

    # Grid :
    pos = np.zeros((n_dim, xcells, ycells))

        # grid with pos[0] for the 1st dimension and pos[1] for the second
    lstx = np.linspace(xdim[0], xdim[1], xcells)
    lsty = np.linspace(ydim[0], ydim[1], ycells)
    dx = lstx[1] - lstx[0]
    dy = lsty[1] - lsty[0]

    pos[0] = lstx[:, None]
    pos[1] = lsty[None, :]

    # Gravity :
    phi = 0
    g = 0

    # Density :
        #masks
    mask1 = (0 <= pos[1,:,:]) & (pos[1,:,:] <= 1/4)
    mask2 = (1/4 <= pos[1,:,:]) & (pos[1,:,:] <= 1/2)
    mask3 = (1/2 <= pos[1,:,:]) & (pos[1,:,:] <= 3/4)
    mask4 = (3/4 <= pos[1,:,:]) & (pos[1,:,:] <= 1)

        #variables
    rho_1 = 1
    rho_2 = 2
    rho_m = (rho_1-rho_2)/2
    L = 0.025

        #rho
    rho = np.zeros((xcells, ycells))
    rho[mask1] = rho_1 - rho_m * np.exp((pos[1,:,:][mask1]-1/4)/L)
    rho[mask2] = rho_2 + rho_m * np.exp((-pos[1,:,:][mask2]+1/4)/L)
    rho[mask3] = rho_2 + rho_m * np.exp(-(3/4 - pos[1,:,:][mask3])/L)
    rho[mask4] = rho_1 - rho_m * np.exp(-(pos[1,:,:][mask4] - 3/4)/L)

    # Pressure
    P = P0
    
    # Sound speed
    c_s = np.sqrt((adiab_ind*P)/rho)

    # Velocity field:
        #variables
    U_1 = 0.5
    U_2 = -0.5
    U_m = (U_1 - U_2)/2
    L = 0.025

        #velocity
    vel = np.zeros((n_dim, xcells, ycells))
    vel[0,:,:][mask1] = U_1 - U_m * np.exp((pos[1,:,:][mask1]-1/4)/L)
    vel[0,:,:][mask2] = U_2 + U_m * np.exp((-pos[1,:,:][mask2]+1/4)/L)
    vel[0,:,:][mask3] = U_2 + U_m * np.exp(-(3/4 - pos[1,:,:][mask3])/L)
    vel[0,:,:][mask4] = U_1 - U_m * np.exp(-(pos[1,:,:][mask4] - 3/4)/L)

    vel[1,:,:] = 0.01 * np.sin(4*np.pi*pos[0,:,:])

    # Internal energy and energy
    Eint = P/(rho*(adiab_ind - 1))
    E = rho * ((vel[0]**2 + vel[1]**2)/2 + Eint + phi)
    return pos, g, phi, rho, P, E, Eint, c_s, vel, lstx, lsty, dx, dy


def ini_U(rho, vel, E, xcells, ycells):
    """For a given density, velocity and internal energy, returns the initial
    values of U"""
    U = np.zeros((4, xcells, ycells))
    U[0,:,:] = rho
    U[1,:,:] = rho*vel[0]
    U[2,:,:] = rho*vel[1]
    U[3,:,:] = E
    return U
