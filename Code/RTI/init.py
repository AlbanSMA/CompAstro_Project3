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

    pos[0] = np.outer(lstx, np.ones(ycells)).reshape(xcells, ycells)
    pos[1] = np.outer(np.ones(xcells), lsty).reshape(xcells, ycells)

    # Gravity :
    g = -0.1*pos[1]

    # Density :
    mask1 = (pos[1,:,:] < 0)
    mask2 = (pos[1,:,:] > 0)

    rho = np.zeros((xcells, ycells))
    rho[mask1] = 1
    rho[mask2] = 2

    # Pressure
    P = P0 - g*rho

    # Internal energy
    Eint = P/(rho*(adiab_ind - 1))
    
    # Sound speed
    c_s = np.sqrt((adiab_ind*P)/rho)

    # Velocity field:
    vel = np.zeros((n_dim, xcells, ycells))
    vel[1] = (0.01*(1 + np.cos(4*np.pi*pos[0])) * (1 + np.cos(3*np.pi*pos[1])))/4
    return pos, g, rho, P, Eint, c_s, vel, lstx, lsty, dx, dy


def ini_U(rho, vel, Eint, xcells, ycells, n_dim):
    """For a given density, velocity and internal energy, returns the initial
    values of U"""
    U = np.zeros((n_dim, 3, xcells, ycells))
    U[:,0,:,:] = rho
    U[:,1,:,:] = rho*vel
    U[:,2,:,:] = Eint
    return U
