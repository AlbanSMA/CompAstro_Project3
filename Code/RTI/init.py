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
    phi = 0.1*pos[1]
    g = -0.1

    # Density :
    mask1 = (pos[1,:,:] < 0)
    mask2 = (pos[1,:,:] > 0)

    rho = np.zeros((xcells, ycells))
    rho[mask1] = 0.5
    rho[mask2] = 5

    # Pressure
    P = P0 - phi*rho
    
    # Sound speed
    c_s = np.sqrt((adiab_ind*P)/rho)

    # Velocity field:
    vel = np.zeros((n_dim, xcells, ycells))
    vel[1] = 0.1 * (1 + np.cos(4*np.pi*pos[0])) * (1 + np.cos(3*np.pi*pos[1]))/4

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
