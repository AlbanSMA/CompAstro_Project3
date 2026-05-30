import numpy as np

def getDensity(rho, n_dim, xcells, ycells, dx, dy):
    """Given the current densities, the shape of the grid, and the intervals in x and y,
    returns two matrices of the size of the grid containing the densities on the left and
    right of the interfaces, the first dimension of each being of size two for x and y."""
    rhoL, rhoR, rhoU, rhoD = np.zeros((xcells, ycells)), np.zeros((xcells, ycells)), np.zeros((xcells, ycells)), np.zeros((xcells, ycells))

    #Determine the slope of the density at the interface
    # Using the minmod method : 
        # get list of rho at m-1
    rhomm1 = np.zeros((n_dim, xcells, ycells))
            # y direction
    rhomm1[1] = np.roll(rho, -1, axis=0)

            # x direction
    rhomm1[0] = np.roll(rho, -1, axis=1)

        # list of rho at m+1
    rhomp1 = np.zeros((n_dim, xcells, ycells))
            # y direction
    rhomp1[1] = np.roll(rho, 1, axis=0)
            # x direction
    rhomp1[0] = np.roll(rho, 1, axis=1)

        # Then get the slopes (rho[0] == rho[1])
    pluslope, minslope = np.zeros((n_dim, xcells, ycells)), np.zeros((n_dim, xcells, ycells))
    minslope[1] = (rho - rhomm1[1])
    pluslope[1] = (rhomp1[1] - rho)

    minslope[0] = (rho - rhomm1[0])
    pluslope[0] = (rhomp1[0] - rho)

        # Get the signes of the slopes
    signmin = np.sign(minslope)
    signplu = np.sign(pluslope)

        # Finally, obtain the slope
    masksamesign = (signmin == signplu)
    maskdiffsign = (signmin != signplu)

    slope = np.zeros((n_dim, xcells, ycells))
            # this method is only used if they have the same signs
    slope[masksamesign] = signmin[masksamesign]*np.minimum(np.abs(minslope[masksamesign]), np.abs(pluslope[masksamesign]))
    slope[0][maskdiffsign[0]] = (rhomp1[0][maskdiffsign[0]] - rhomm1[0][maskdiffsign[0]])/2
    slope[1][maskdiffsign[1]] = (rhomp1[1][maskdiffsign[1]] - rhomm1[1][maskdiffsign[1]])/2

    rhoD = rho + 0.5*slope[0]
    rhoL = rho + 0.5*slope[1]

    rhoU = rho - 0.5*slope[0]
    rhoR = rho - 0.5*slope[1]

    return rhoL, rhoR, rhoD, rhoU

def getVelocities(vel, n_dim, xcells, ycells, dx, dy):
    """Given the current velocities, the shape of the grid, and the intervals in x and y,
    returns two matrices of the size of the grid containing the velocities on the left and
    right of the interfaces, the first dimension of each being of size two for x and y."""
    velL, velR, velD, velU = np.zeros((xcells, ycells)), np.zeros((xcells, ycells)), np.zeros((xcells, ycells)), np.zeros((xcells, ycells))

    #Determine the slope of the velocity at the interface
    # Using the minmod method : 
        # get list of vel at m-1
    velmm1 = np.zeros((n_dim, xcells, ycells))
            
            # y direction
    velmm1[1] = np.roll(vel, -1, axis=0)
            # x direction
    velmm1[0] = np.roll(vel, -1, axis=1)

        # list of vel at m+1
    velmp1 = np.zeros((n_dim, xcells, ycells))
            # y direction
    velmp1[1] = np.roll(vel, 1, axis=0)
            # x direction
    velmp1[0] = np.roll(vel, 1, axis=1)

        # Then get the slopes
    pluslope, minslope = np.zeros((n_dim, xcells, ycells)), np.zeros((n_dim, xcells, ycells))
    minslope[0] = (vel - velmm1[0])
    pluslope[0] = (velmp1[0] - vel)

    minslope[1] = (vel - velmm1[1])
    pluslope[1] = (velmp1[1] - vel)

        # Get the signes of the slopes
    signmin = np.sign(minslope)
    signplu = np.sign(pluslope)

        # Finally, obtain the slope
    masksamesign = (signmin == signplu)
    maskdiffsign = (signmin != signplu)

    slope = np.zeros((n_dim, xcells, ycells))
            # but this method only works if they have the same signs
    slope[masksamesign] = signmin[masksamesign]*np.minimum(np.abs(minslope[masksamesign]), np.abs(pluslope[masksamesign]))
    slope[0][maskdiffsign[0]] = (velmp1[0][maskdiffsign[0]] - velmm1[0][maskdiffsign[0]])/2
    slope[1][maskdiffsign[1]] = (velmp1[1][maskdiffsign[1]] - velmm1[1][maskdiffsign[1]])/2

    velL = vel + 0.5*slope[0]
    velD = vel + 0.5*slope[1]

    velR = vel - 0.5*slope[0]
    velU = vel - 0.5*slope[1]
    return velL, velR, velD, velU

def getEnergies(Eint, n_dim, xcells, ycells, dx, dy):
    """Given the current energies, the shape of the grid, and the intervals in x and y,
    returns two matrices of the size of the grid containing the energies on the left and
    right of the interfaces, the first dimension of each being of size two for x and y."""
    EintL, EintR, EintU, EintD = np.zeros((xcells, ycells)), np.zeros((xcells, ycells)), np.zeros((xcells, ycells)), np.zeros((xcells, ycells))

    #Determine the slope of the density at the interface
    # Using the minmod method : 
        # get list of Eint at m-1
    Eintmm1 = np.zeros((n_dim, xcells, ycells))
            # y direction
    Eintmm1[1] = np.roll(Eint, -1, axis=0)

            # x direction
    Eintmm1[0] = np.roll(Eint, -1, axis=1)

        # list of Eint at m+1
    Eintmp1 = np.zeros((n_dim, xcells, ycells))
            # y direction
    Eintmp1[1] = np.roll(Eint, 1, axis=0)
            # x direction
    Eintmp1[0] = np.roll(Eint, 1, axis=1)

        # Then get the slopes (Eint[0] == Eint[1])
    pluslope, minslope = np.zeros((n_dim, xcells, ycells)), np.zeros((n_dim, xcells, ycells))
    minslope[1] = (Eint - Eintmm1[1])
    pluslope[1] = (Eintmp1[1] - Eint)

    minslope[0] = (Eint - Eintmm1[0])
    pluslope[0] = (Eintmp1[0] - Eint)

        # Get the signes of the slopes
    signmin = np.sign(minslope)
    signplu = np.sign(pluslope)

        # Finally, obtain the slope
    masksamesign = (signmin == signplu)
    maskdiffsign = (signmin != signplu)

    slope = np.zeros((n_dim, xcells, ycells))
            # this method is only used if they have the same signs
    slope[masksamesign] = signmin[masksamesign]*np.minimum(np.abs(minslope[masksamesign]), np.abs(pluslope[masksamesign]))
    slope[0][maskdiffsign[0]] = (Eintmp1[0][maskdiffsign[0]] - Eintmm1[0][maskdiffsign[0]])/2
    slope[1][maskdiffsign[1]] = (Eintmp1[1][maskdiffsign[1]] - Eintmm1[1][maskdiffsign[1]])/2

    EintD = Eint + 0.5*slope[0]
    EintL = Eint + 0.5*slope[1]

    EintU = Eint - 0.5*slope[0]
    EintR = Eint - 0.5*slope[1]
    return EintL, EintR, EintD, EintU
