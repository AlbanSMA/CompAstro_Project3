import numpy as np

def getFluxes(vel, velL, velR, velU, velD, rhoL, rhoR, rhoU, rhoD, EintL, EintR, EintU, EintD, PL, PR, PU, PD, xcells, ycells):
    """Given the values of the primitives at the interfaces, the velocities and 
    the shape of the grid, returns the fluxes at the interfaces"""
    FminL = np.zeros((3, xcells, ycells))
    FminD = np.zeros((3, xcells, ycells))
    FminR = np.zeros((3, xcells, ycells))
    FminU = np.zeros((3, xcells, ycells))

    FplusL = np.zeros((3, xcells, ycells))
    FplusR = np.zeros((3, xcells, ycells))
    FplusD = np.zeros((3, xcells, ycells))
    FplusU = np.zeros((3, xcells, ycells))

    #masks to avoid loops, depending on the velocity:
    maskminvely = vel[0] < 0
    maskminvelx = vel[1] < 0
    maskpluvely = vel[0] >= 0
    maskpluvelx = vel[1] >= 0

    #On the left
    # y direction
        #negative velocity
    FminD[0, :-1, :][maskminvely[1:, :]] = velD[1:, :][maskminvely[1:, :]]**2 * rhoD[1:, :][maskminvely[1:, :]]
    FminD[1, :-1, :][maskminvely[1:, :]] = velD[1:, :][maskminvely[1:, :]]**2 * rhoD[1:, :][maskminvely[1:, :]] + PD[1:, :][maskminvely[1:, :]]
    FminD[2, :-1, :][maskminvely[1:, :]] = velD[1:, :][maskminvely[1:, :]] * (EintD[1:, :][maskminvely[1:, :]] + PD[1:, :][maskminvely[1:, :]])
    
    FplusD[0, :, :][maskminvely] = velD[:, :][maskminvely]**2 * rhoD[:, :][maskminvely]
    FplusD[1, :, :][maskminvely] = velD[:, :][maskminvely]**2 * rhoD[:, :][maskminvely] + PD[:, :][maskminvely]
    FplusD[2, :, :][maskminvely] = velD[:, :][maskminvely] * (EintD[:, :][maskminvely] + PD[:, :][maskminvely])

        #positive velocity
    FminD[0, :, :][maskpluvely] = velD[:, :][maskpluvely]**2 * rhoD[:, :][maskpluvely]
    FminD[1, :, :][maskpluvely] = velD[:, :][maskpluvely]**2 * rhoD[:, :][maskpluvely] + PD[:, :][maskpluvely]
    FminD[2, :, :][maskpluvely] = velD[:, :][maskpluvely] * (EintD[:, :][maskpluvely] + PD[:, :][maskpluvely])

    FplusD[0, 1:, :][maskpluvely[:-1, :]] = velD[:-1, :][maskpluvely[:-1, :]]**2 * rhoD[:-1, :][maskpluvely[:-1, :]]
    FplusD[1, 1:, :][maskpluvely[:-1, :]] = velD[:-1, :][maskpluvely[:-1, :]]**2 * rhoD[:-1, :][maskpluvely[:-1, :]] + PD[:-1, :][maskpluvely[:-1, :]]
    FplusD[2, 1:, :][maskpluvely[:-1, :]] = velD[:-1, :][maskpluvely[:-1, :]] * (EintD[:-1, :][maskpluvely[:-1, :]] + PD[:-1, :][maskpluvely[:-1, :]])

    # y direction
        # negative velocity
    FminL[0, :, :-1][maskminvelx[:, 1:]] = velL[:, 1:][maskminvelx[:, 1:]]**2 * rhoL[:, 1:][maskminvelx[:, 1:]]
    FminL[1, :, :-1][maskminvelx[:, 1:]] = velL[:, 1:][maskminvelx[:, 1:]]**2 * rhoL[:, 1:][maskminvelx[:, 1:]] + PL[:, 1:][maskminvelx[:, 1:]]
    FminL[2, :, :-1][maskminvelx[:, 1:]] = velL[:, 1:][maskminvelx[:, 1:]] * (EintL[:, 1:][maskminvelx[:, 1:]] + PL[:, 1:][maskminvelx[:, 1:]])

    FplusL[0, :, :][maskminvelx] = velL[:, :][maskminvelx]**2 * rhoL[:, :][maskminvelx]
    FplusL[1, :, :][maskminvelx] = velL[:, :][maskminvelx]**2 * rhoL[:, :][maskminvelx] + PL[:, :][maskminvelx]
    FplusL[2, :, :][maskminvelx] = velL[:, :][maskminvelx] * (EintL[:, :][maskminvelx] + PL[:, :][maskminvelx])

        # positive velocity
    FminL[0, :, :][maskpluvelx] = velL[:, :][maskpluvelx]**2 * rhoL[:, :][maskpluvelx]
    FminL[1, :, :][maskpluvelx] = velL[:, :][maskpluvelx]**2 * rhoL[:, :][maskpluvelx] + PL[:, :][maskpluvelx]
    FminL[2, :, :][maskpluvelx] = velL[:, :][maskpluvelx] * (EintL[:, :][maskpluvelx] + PL[:, :][maskpluvelx])

    FplusL[0, :, 1:][maskpluvelx[:, :-1]] = velL[:, :-1][maskpluvelx[:, :-1]]**2 * rhoL[:, :-1][maskpluvelx[:, :-1]]
    FplusL[1, :, 1:][maskpluvelx[:, :-1]] = velL[:, :-1][maskpluvelx[:, :-1]]**2 * rhoL[:, :-1][maskpluvelx[:, :-1]] + PL[:, :-1][maskpluvelx[:, :-1]]
    FplusL[2, :, 1:][maskpluvelx[:, :-1]] = velL[:, :-1][maskpluvelx[:, :-1]] * (EintL[:, :-1][maskpluvelx[:, :-1]] + PL[:, :-1][maskpluvelx[:, :-1]])


    #On the right:
    # x direction
        #negative velocity
    FminU[0, :-1, :][maskminvely[1:, :]] = velU[1:, :][maskminvely[1:, :]]**2 * rhoU[ 1:, :][maskminvely[1:, :]]
    FminU[1, :-1, :][maskminvely[1:, :]] = velU[1:, :][maskminvely[1:, :]]**2 * rhoU[ 1:, :][maskminvely[1:, :]] + PU[1:, :][maskminvely[1:, :]]
    FminU[2, :-1, :][maskminvely[1:, :]] = velU[1:, :][maskminvely[1:, :]] * (EintU[1:, :][maskminvely[1:, :]] + PU[1:, :][maskminvely[1:, :]])
    
    FplusU[0, :, :][maskminvely] = velU[:, :][maskminvely]**2 * rhoU[ :, :][maskminvely]
    FplusU[1, :, :][maskminvely] = velU[:, :][maskminvely]**2 * rhoU[ :, :][maskminvely] + PU[:, :][maskminvely]
    FplusU[2, :, :][maskminvely] = velU[:, :][maskminvely] * (EintU[:, :][maskminvely] + PU[:, :][maskminvely])

        #positive velocity
    FminU[0, :, :][maskpluvely] = velU[:, :][maskpluvely]**2 * rhoU[ :, :][maskpluvely]
    FminU[1, :, :][maskpluvely] = velU[:, :][maskpluvely]**2 * rhoU[ :, :][maskpluvely] + PU[:, :][maskpluvely]
    FminU[2, :, :][maskpluvely] = velU[:, :][maskpluvely] * (EintU[:, :][maskpluvely] + PU[:, :][maskpluvely])

    FplusU[0, 1:, :][maskpluvely[:-1, :]] = velU[:-1, :][maskpluvely[:-1, :]]**2 * rhoU[ :-1, :][maskpluvely[:-1, :]]
    FplusU[1, 1:, :][maskpluvely[:-1, :]] = velU[:-1, :][maskpluvely[:-1, :]]**2 * rhoU[ :-1, :][maskpluvely[:-1, :]] + PU[:-1, :][maskpluvely[:-1, :]]
    FplusU[2, 1:, :][maskpluvely[:-1, :]] = velU[:-1, :][maskpluvely[:-1, :]] * (EintU[:-1, :][maskpluvely[:-1, :]] + PU[:-1, :][maskpluvely[:-1, :]])

    # y direction
        # negative velocity
    FminR[0, :, :-1][maskminvelx[:, 1:]] = velR[:, 1:][maskminvelx[:, 1:]]**2 * rhoR[ :, 1:][maskminvelx[:, 1:]]
    FminR[1, :, :-1][maskminvelx[:, 1:]] = velR[:, 1:][maskminvelx[:, 1:]]**2 * rhoR[ :, 1:][maskminvelx[:, 1:]] + PR[:, 1:][maskminvelx[:, 1:]]
    FminR[2, :, :-1][maskminvelx[:, 1:]] = velR[:, 1:][maskminvelx[:, 1:]] * (EintR[:, 1:][maskminvelx[:, 1:]] + PR[:, 1:][maskminvelx[:, 1:]])

    FplusR[0, :, :][maskminvelx] = velR[:, :][maskminvelx]**2 * rhoR[ :, :][maskminvelx]
    FplusR[1, :, :][maskminvelx] = velR[:, :][maskminvelx]**2 * rhoR[ :, :][maskminvelx] + PR[:, :][maskminvelx]
    FplusR[2, :, :][maskminvelx] = velR[:, :][maskminvelx] * (EintR[:, :][maskminvelx] + PR[:, :][maskminvelx])

        # positive velocity
    FminR[0, :, :][maskpluvelx] = velR[:, :][maskpluvelx]**2 * rhoR[ :, :][maskpluvelx]
    FminR[1, :, :][maskpluvelx] = velR[:, :][maskpluvelx]**2 * rhoR[ :, :][maskpluvelx] + PR[:, :][maskpluvelx]
    FminR[2, :, :][maskpluvelx] = velR[:, :][maskpluvelx] * (EintR[:, :][maskpluvelx] + PR[:, :][maskpluvelx])

    FplusR[0, :, 1:][maskpluvelx[:, :-1]] = velR[:, :-1][maskpluvelx[:, :-1]]**2 * rhoR[ :, :-1][maskpluvelx[:, :-1]]
    FplusR[1, :, 1:][maskpluvelx[:, :-1]] = velR[:, :-1][maskpluvelx[:, :-1]]**2 * rhoR[ :, :-1][maskpluvelx[:, :-1]] + PR[:, :-1][maskpluvelx[:, :-1]]
    FplusR[2, :, 1:][maskpluvelx[:, :-1]] = velR[:, :-1][maskpluvelx[:, :-1]] * (EintR[:, :-1][maskpluvelx[:, :-1]] + PR[:, :-1][maskpluvelx[:, :-1]])


    #Then deal with the extrema at 0,0; 0,-1; -1,0 and -1,-1 for both matrices 
        #negative velocity, x :
    FminD[0, -1, :][maskminvely[-1, :]] = velD[-1, :][maskminvely[-1, :]]**2 * rhoD[-1, :][maskminvely[-1, :]]
    FminD[1, -1, :][maskminvely[-1, :]] = velD[-1, :][maskminvely[-1, :]]**2 * rhoD[-1, :][maskminvely[-1, :]] + PD[-1, :][maskminvely[-1, :]]
    FminD[2, -1, :][maskminvely[-1, :]] = velD[-1, :][maskminvely[-1, :]] * (EintD[-1, :][maskminvely[-1, :]] + PD[-1, :][maskminvely[-1, :]])

    FminU[0, -1, :][maskminvely[-1, :]] = velU[-1, :][maskminvely[-1, :]]**2 * rhoU[-1, :][maskminvely[-1, :]]
    FminU[1, -1, :][maskminvely[-1, :]] = velU[-1, :][maskminvely[-1, :]]**2 * rhoU[-1, :][maskminvely[-1, :]] + PU[-1, :][maskminvely[-1, :]]
    FminU[2, -1, :][maskminvely[-1, :]] = velU[-1, :][maskminvely[-1, :]] * (EintU[-1, :][maskminvely[-1, :]] + PU[-1, :][maskminvely[-1, :]])

        #positive velocity, x
    FplusD[0, 0, :][maskpluvely[0, :]] = velD[0, :][maskpluvely[0, :]]**2 * rhoD[0, :][maskpluvely[0, :]]
    FplusD[1, 0, :][maskpluvely[0, :]] = velD[0, :][maskpluvely[0, :]]**2 * rhoD[0, :][maskpluvely[0, :]] + PD[0, :][maskpluvely[0, :]]
    FplusD[2, 0, :][maskpluvely[0, :]] = velD[0, :][maskpluvely[0, :]] * (EintD[0, :][maskpluvely[0, :]] + PD[0, :][maskpluvely[0, :]])

    FplusU[0, 0, :][maskpluvely[0, :]] = velU[0, :][maskpluvely[0, :]]**2 * rhoU[0, :][maskpluvely[0, :]]
    FplusU[1, 0, :][maskpluvely[0, :]] = velU[0, :][maskpluvely[0, :]]**2 * rhoU[0, :][maskpluvely[0, :]] + PU[0, :][maskpluvely[0, :]]
    FplusU[2, 0, :][maskpluvely[0, :]] = velU[0, :][maskpluvely[0, :]] * (EintU[0, :][maskpluvely[0, :]] + PU[0, :][maskpluvely[0, :]])

        #negative velocity, y :
    FminL[0, :, -1][maskminvelx[:, -1]] = velL[:, -1][maskminvelx[:, -1]]**2 * rhoL[:, -1][maskminvelx[:, -1]]
    FminL[1, :, -1][maskminvelx[:, -1]] = velL[:, -1][maskminvelx[:, -1]]**2 * rhoL[:, -1][maskminvelx[:, -1]] + PL[:, -1][maskminvelx[:, -1]]
    FminL[2, :, -1][maskminvelx[:, -1]] = velL[:, -1][maskminvelx[:, -1]] * (EintL[:, -1][maskminvelx[:, -1]] + PL[:, -1][maskminvelx[:, -1]])

    FminR[0, :, -1][maskminvelx[:, -1]] = velR[:, -1][maskminvelx[:, -1]]**2 * rhoR[:, -1][maskminvelx[:, -1]]
    FminR[1, :, -1][maskminvelx[:, -1]] = velR[:, -1][maskminvelx[:, -1]]**2 * rhoR[:, -1][maskminvelx[:, -1]] + PR[:, -1][maskminvelx[:, -1]]
    FminR[2, :, -1][maskminvelx[:, -1]] = velR[:, -1][maskminvelx[:, -1]] * (EintR[:, -1][maskminvelx[:, -1]] + PR[:, -1][maskminvelx[:, -1]])

        #positive velocity, y:
    FplusL[0, :, 0][maskpluvelx[:, 0]] = velL[:, 0][maskpluvelx[:, 0]]**2 * rhoL[:, 0][maskpluvelx[:, 0]]
    FplusL[1, :, 0][maskpluvelx[:, 0]] = velL[:, 0][maskpluvelx[:, 0]]**2 * rhoL[:, 0][maskpluvelx[:, 0]] + PL[:, 0][maskpluvelx[:, 0]]
    FplusL[2, :, 0][maskpluvelx[:, 0]] = velL[:, 0][maskpluvelx[:, 0]] * (EintL[:, 0][maskpluvelx[:, 0]] + PL[:, 0][maskpluvelx[:, 0]])

    FplusR[0, :, 0][maskpluvelx[:, 0]] = velR[:, 0][maskpluvelx[:, 0]]**2 * rhoR[ :, 0][maskpluvelx[:, 0]]
    FplusR[1, :, 0][maskpluvelx[:, 0]] = velR[:, 0][maskpluvelx[:, 0]]**2 * rhoR[ :, 0][maskpluvelx[:, 0]] + PR[:, 0][maskpluvelx[:, 0]]
    FplusR[2, :, 0][maskpluvelx[:, 0]] = velR[:, 0][maskpluvelx[:, 0]] * (EintR[:, 0][maskpluvelx[:, 0]] + PR[:, 0][maskpluvelx[:, 0]])

    return FplusL, FplusR, FminL, FminR, FplusD, FplusU, FminD, FminU

############################################################################################""
def getUs(vel, velL, velR, velU, velD, rhoL, rhoR, rhoU, rhoD, EintL, EintR, EintU, EintD, PL, PR, PU, PD, xcells, ycells):
    """Given the primitive variables, return the concerved variables U"""
    
    UminL = np.zeros((3, xcells, ycells))
    UminR = np.zeros((3, xcells, ycells))
    UplusL = np.zeros((3, xcells, ycells))
    UplusR = np.zeros((3, xcells, ycells))

    UminD = np.zeros((3, xcells, ycells))
    UminU = np.zeros((3, xcells, ycells))
    UplusD = np.zeros((3, xcells, ycells))
    UplusU = np.zeros((3, xcells, ycells))

    #masks to avoid loops:
    maskminvelx = vel[1] < 0
    maskminvely = vel[0] < 0
    maskpluvelx = vel[1] >= 0
    maskpluvely = vel[0] >= 0

    #On the left
    # x direction
        #negative velocity
    UminD[0, :-1, :][maskminvely[1:, :]] = rhoD[1:, :][maskminvely[1:, :]]
    UminD[1, :-1, :][maskminvely[1:, :]] = velD[1:, :][maskminvely[1:, :]] * rhoD[1:, :][maskminvely[1:, :]]
    UminD[2, :-1, :][maskminvely[1:, :]] = EintD[1:, :][maskminvely[1:, :]]
    
    UplusD[0, :, :][maskminvely] = rhoD[:, :][maskminvely]
    UplusD[1, :, :][maskminvely] = velD[:, :][maskminvely] * rhoD[:, :][maskminvely]
    UplusD[2, :, :][maskminvely] = EintD[:, :][maskminvely]

        #positive velocity
    UminD[0, :, :][maskpluvely] = rhoD[:, :][maskpluvely]
    UminD[1, :, :][maskpluvely] = velD[:, :][maskpluvely] * rhoD[:, :][maskpluvely]
    UminD[2, :, :][maskpluvely] = EintD[:, :][maskpluvely]

    UplusD[0, 1:, :][maskpluvely[:-1, :]] = rhoD[:-1, :][maskpluvely[:-1, :]]
    UplusD[1, 1:, :][maskpluvely[:-1, :]] = velD[:-1, :][maskpluvely[:-1, :]] * rhoD[:-1, :][maskpluvely[:-1, :]]
    UplusD[2, 1:, :][maskpluvely[:-1, :]] = EintD[:-1, :][maskpluvely[:-1, :]]

    # y direction
        # negative velocity
    UminL[0, :, :-1][maskminvelx[:, 1:]] = rhoL[:, 1:][maskminvelx[:, 1:]]
    UminL[1, :, :-1][maskminvelx[:, 1:]] = velL[:, 1:][maskminvelx[:, 1:]] * rhoL[:, 1:][maskminvelx[:, 1:]]
    UminL[2, :, :-1][maskminvelx[:, 1:]] = EintL[:, 1:][maskminvelx[:, 1:]]

    UplusL[0, :, :][maskminvelx] = rhoL[:, :][maskminvelx]
    UplusL[1, :, :][maskminvelx] = velL[:, :][maskminvelx] * rhoL[:, :][maskminvelx]
    UplusL[2, :, :][maskminvelx] = EintL[:, :][maskminvelx]

        # positive velocity
    UminL[0, :, :][maskpluvelx] = rhoL[:, :][maskpluvelx]
    UminL[1, :, :][maskpluvelx] = velL[:, :][maskpluvelx] * rhoL[:, :][maskpluvelx]
    UminL[2, :, :][maskpluvelx] = EintL[:, :][maskpluvelx]

    UplusL[0, :, 1:][maskpluvelx[:, :-1]] = rhoL[:, :-1][maskpluvelx[:, :-1]]
    UplusL[1, :, 1:][maskpluvelx[:, :-1]] = velL[:, :-1][maskpluvelx[:, :-1]] * rhoL[:, :-1][maskpluvelx[:, :-1]]
    UplusL[2, :, 1:][maskpluvelx[:, :-1]] = EintL[:, :-1][maskpluvelx[:, :-1]]


    #On the right:
    # x direction
        #negative velocity
    UminU[0, :-1, :][maskminvely[1:, :]] = rhoU[ 1:, :][maskminvely[1:, :]]
    UminU[1, :-1, :][maskminvely[1:, :]] = velU[1:, :][maskminvely[1:, :]] * rhoU[ 1:, :][maskminvely[1:, :]]
    UminU[2, :-1, :][maskminvely[1:, :]] = EintU[1:, :][maskminvely[1:, :]]
    
    UplusU[0, :, :][maskminvely] = rhoU[ :, :][maskminvely]
    UplusU[1, :, :][maskminvely] = velU[:, :][maskminvely] * rhoU[ :, :][maskminvely]
    UplusU[2, :, :][maskminvely] = EintU[:, :][maskminvely]

        #positive velocity
    UminU[0, :, :][maskpluvely] = rhoU[ :, :][maskpluvely]
    UminU[1, :, :][maskpluvely] = velU[:][maskpluvely] * rhoU[ :, :][maskpluvely]
    UminU[2, :, :][maskpluvely] = EintU[:, :][maskpluvely]

    UplusU[0, 1:, :][maskpluvely[:-1, :]] = rhoU[ :-1, :][maskpluvely[:-1, :]]
    UplusU[1, 1:, :][maskpluvely[:-1, :]] = velU[:-1, :][maskpluvely[:-1, :]] * rhoU[ :-1, :][maskpluvely[:-1, :]]
    UplusU[2, 1:, :][maskpluvely[:-1, :]] = EintU[:-1, :][maskpluvely[:-1, :]]

    # y direction
        # negative velocity
    UminR[0, :, :-1][maskminvelx[:, 1:]] = rhoR[ :, 1:][maskminvelx[:, 1:]]
    UminR[1, :, :-1][maskminvelx[:, 1:]] = velR[:, 1:][maskminvelx[:, 1:]] * rhoR[ :, 1:][maskminvelx[:, 1:]]
    UminR[2, :, :-1][maskminvelx[:, 1:]] = EintR[:, 1:][maskminvelx[:, 1:]]

    UplusR[0, :, :][maskminvelx] = rhoR[ :, :][maskminvelx]
    UplusR[1, :, :][maskminvelx] = velR[:, :][maskminvelx] * rhoR[ :, :][maskminvelx]
    UplusR[2, :, :][maskminvelx] = EintR[:, :][maskminvelx]

        # positive velocity
    UminR[0, :, :][maskpluvelx] = rhoR[ :, :][maskpluvelx]
    UminR[1, :, :][maskpluvelx] = velR[:, :][maskpluvelx] * rhoR[ :, :][maskpluvelx]
    UminR[2, :, :][maskpluvelx] = EintR[:, :][maskpluvelx]

    UplusR[0, :, 1:][maskpluvelx[:, :-1]] = rhoR[ :, :-1][maskpluvelx[:, :-1]]
    UplusR[1, :, 1:][maskpluvelx[:, :-1]] = velR[:, :-1][maskpluvelx[:, :-1]] * rhoR[ :, :-1][maskpluvelx[:, :-1]]
    UplusR[2, :, 1:][maskpluvelx[:, :-1]] = EintR[:, :-1][maskpluvelx[:, :-1]]


    #Then deal with the extrema at 0 and -1 for both matrices 
        #negative velocity, x :
    UminD[0, -1, :][maskminvely[-1, :]] = rhoD[-1, :][maskminvely[-1, :]]
    UminD[1, -1, :][maskminvely[-1, :]] = velD[-1, :][maskminvely[-1, :]] * rhoD[-1, :][maskminvely[-1, :]]
    UminD[2, -1, :][maskminvely[-1, :]] = EintD[-1, :][maskminvely[-1, :]]

    UminU[0, -1, :][maskminvely[-1, :]] = rhoU[-1, :][maskminvely[-1, :]]
    UminU[1, -1, :][maskminvely[-1, :]] = velU[-1, :][maskminvely[-1, :]] * rhoU[-1, :][maskminvely[-1, :]]
    UminU[2, -1, :][maskminvely[-1, :]] = EintU[-1, :][maskminvely[-1, :]]

        #positive velocity, x
    UplusD[0, 0, :][maskpluvely[0, :]] = rhoD[0, :][maskpluvely[0, :]]
    UplusD[1, 0, :][maskpluvely[0, :]] = velD[0, :][maskpluvely[0, :]] * rhoD[0, :][maskpluvely[0, :]]
    UplusD[2, 0, :][maskpluvely[0, :]] = EintD[0, :][maskpluvely[0, :]]

    UplusU[0, 0, :][maskpluvely[0, :]] = rhoU[0, :][maskpluvely[0, :]]
    UplusU[1, 0, :][maskpluvely[0, :]] = velU[0, :][maskpluvely[0, :]] * rhoU[0, :][maskpluvely[0, :]]
    UplusU[2, 0, :][maskpluvely[0, :]] = EintU[0, :][maskpluvely[0, :]]

        #negative velocity, y :
    UminL[0, :, -1][maskminvelx[:, -1]] = rhoL[:, -1][maskminvelx[:, -1]]
    UminL[1, :, -1][maskminvelx[:, -1]] = velL[:, -1][maskminvelx[:, -1]] * rhoL[:, -1][maskminvelx[:, -1]]
    UminL[2, :, -1][maskminvelx[:, -1]] = EintL[:, -1][maskminvelx[:, -1]]

    UminR[0, :, -1][maskminvelx[:, -1]] = rhoR[:, -1][maskminvelx[:, -1]]
    UminR[1, :, -1][maskminvelx[:, -1]] = velR[:, -1][maskminvelx[:, -1]] * rhoR[:, -1][maskminvelx[:, -1]]
    UminR[2, :, -1][maskminvelx[:, -1]] = EintR[:, -1][maskminvelx[:, -1]]

        #positive velocity, y:
    UplusL[0, :, 0][maskpluvelx[:, 0]] = rhoL[:, 0][maskpluvelx[:, 0]]
    UplusL[1, :, 0][maskpluvelx[:, 0]] = velL[:, 0][maskpluvelx[:, 0]] * rhoL[:, 0][maskpluvelx[:, 0]]
    UplusL[2, :, 0][maskpluvelx[:, 0]] = EintL[:, 0][maskpluvelx[:, 0]]

    UplusR[0, :, 0][maskpluvelx[:, 0]] = rhoR[ :, 0][maskpluvelx[:, 0]]
    UplusR[1, :, 0][maskpluvelx[:, 0]] = velR[:, 0][maskpluvelx[:, 0]] * rhoR[:, 0][maskpluvelx[:, 0]]
    UplusR[2, :, 0][maskpluvelx[:, 0]] = EintR[:, 0][maskpluvelx[:, 0]]

    return UplusL, UplusR, UminL, UminR, UplusD, UplusU, UminD, UminU