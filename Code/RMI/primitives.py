import numpy as np

def getPrimitive(var, n_dim, xcells, ycells):
    """Given the current variables, the shape of the grid, and the intervals in x and y,
    returns four matrices of the size of the grid containing the variables on the left,
    right, above and bellow the interfaces."""
    varL, varR, varU, varD = np.zeros((xcells, ycells)), np.zeros((xcells, ycells)), np.zeros((xcells, ycells)), np.zeros((xcells, ycells))

    #Determine the slope of the variable at the interface
    # Using the minmod method : 
        # get list of var at m-1
    varmm1 = np.zeros((n_dim, xcells, ycells))
            # y direction
    varmm1[1] = np.roll(var, 1, axis=1)

            # x direction
    varmm1[0] = np.roll(var, 1, axis=0)

        # list of var at m+1
    varmp1 = np.zeros((n_dim, xcells, ycells))
            # y direction
    varmp1[1] = np.roll(var, -1, axis=1)

            # x direction
    varmp1[0] = np.roll(var, -1, axis=0)

        # Then get the slopes
    pluslope, minslope = np.zeros((n_dim, xcells, ycells)), np.zeros((n_dim, xcells, ycells))
    minslope[1] = (var - varmm1[1])
    pluslope[1] = (varmp1[1] - var)

    minslope[0] = (var - varmm1[0])
    pluslope[0] = (varmp1[0] - var)

        # Get the signes of the slopes
    signmin = np.sign(minslope)
    signplu = np.sign(pluslope)

        # Finally, obtain the slope
    masksamesign = (signmin == signplu)

    slope = np.zeros((n_dim, xcells, ycells))
            # the method is different if they have/don't have the same signs
    slope[masksamesign] = signmin[masksamesign]*np.minimum(np.abs(minslope[masksamesign]), np.abs(pluslope[masksamesign]))

    # Within the m cell, 
        # the "down" boundary is the boundary from m+1/2
        # the "up" boundary is the boundary from m-1/2
        # the "left" boundary is the boundary from m+1/2
        # the "right" boundary is the boundary from m-1/2
    varD = var + 0.5*slope[1]
    varL = var + 0.5*slope[0]

    varU = var - 0.5*slope[1]
    varR = var - 0.5*slope[0]

    return varL, varR, varD, varU