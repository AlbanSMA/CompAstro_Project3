import numpy as np

def getFluxes(vel, velLx, velRx, velUx, velDx, velLy, velRy, velUy, velDy, rhoL, rhoR, rhoU, rhoD, EL, ER, EU, ED, PL, PR, PU, PD, xcells, ycells):
    """Given the values of the primitives at the interfaces, the velocities and 
    the shape of the grid, returns the fluxes at the interfaces"""
    FminL = np.zeros((4, xcells, ycells))
    FminD = np.zeros((4, xcells, ycells))
    FminR = np.zeros((4, xcells, ycells))
    FminU = np.zeros((4, xcells, ycells))

    FplusL = np.zeros((4, xcells, ycells))
    FplusR = np.zeros((4, xcells, ycells))
    FplusD = np.zeros((4, xcells, ycells))
    FplusU = np.zeros((4, xcells, ycells))

    #masks to avoid loops, depending on the velocity:
    maskminvely = vel[1] < 0
    maskpluvely = vel[1] >= 0

    maskminvelx = vel[0] < 0
    maskpluvelx = vel[1] >= 0

        # Get all possible structures
    FD0 = rhoD*velDy
    FD1 = rhoD*velDx*velDy
    FD2 = rhoD*velDy*velDy + PD
    FD3 = (ED + PD) * velDy

    FR0 = rhoR*velRx
    FR1 = rhoR*velRx*velRx + PR
    FR2 = rhoR*velRx*velRy
    FR3 = (ER + PR) * velRx

    FL0 = rhoL*velLx
    FL1 = rhoL*velLx*velLx + PL
    FL2 = rhoL*velLx*velLy
    FL3 = (EL + PL) * velLx

    FU0 = rhoU*velUy
    FU1 = rhoU*velUx*velUy
    FU2 = rhoU*velUy*velUy + PU
    FU3 = (EU + PU) * velUx

# down, y direction
        #negative velocity
    FminD[0][maskminvely] = np.roll(FD0, -1, axis=0)[maskminvely]
    FminD[1][maskminvely] = np.roll(FD1, -1, axis=0)[maskminvely]
    FminD[2][maskminvely] = np.roll(FD2, -1, axis=0)[maskminvely]
    FminD[3][maskminvely] = np.roll(FD3, -1, axis=0)[maskminvely]
    
    FplusD[0][maskminvely] = FD0[maskminvely]
    FplusD[1][maskminvely] = FD1[maskminvely]
    FplusD[2][maskminvely] = FD2[maskminvely]
    FplusD[3][maskminvely] = FD3[maskminvely]

        #positive velocity
    FminD[0][maskpluvely] = FD0[maskpluvely]
    FminD[1][maskpluvely] = FD1[maskpluvely]
    FminD[2][maskpluvely] = FD2[maskpluvely]
    FminD[3][maskpluvely] = FD3[maskpluvely]

    FplusD[0][maskpluvely] = np.roll(FD0, 1, axis=0)[maskpluvely]
    FplusD[1][maskpluvely] = np.roll(FD1, 1, axis=0)[maskpluvely]
    FplusD[2][maskpluvely] = np.roll(FD2, 1, axis=0)[maskpluvely]
    FplusD[3][maskpluvely] = np.roll(FD2, 1, axis=0)[maskpluvely]

    # left, x direction
        # negative velocity
    FminL[0][maskminvelx] = np.roll(FL0, -1, axis=1)[maskminvelx]
    FminL[1][maskminvelx] = np.roll(FL1, -1, axis=1)[maskminvelx]
    FminL[2][maskminvelx] = np.roll(FL2, -1, axis=1)[maskminvelx]
    FminL[3][maskminvelx] = np.roll(FL3, -1, axis=1)[maskminvelx]

    FplusL[0][maskminvelx] = FL0[maskminvelx]
    FplusL[1][maskminvelx] = FL1[maskminvelx]
    FplusL[2][maskminvelx] = FL2[maskminvelx]
    FplusL[3][maskminvelx] = FL3[maskminvelx]

        # positive velocity
    FminL[0][maskpluvelx] = FL0[maskpluvelx]
    FminL[1][maskpluvelx] = FL1[maskpluvelx]
    FminL[2][maskpluvelx] = FL2[maskpluvelx]
    FminL[3][maskpluvelx] = FL3[maskpluvelx]

    FplusL[0][maskpluvelx] = np.roll(FL0, 1, axis=1)[maskpluvelx]
    FplusL[1][maskpluvelx] = np.roll(FL1, 1, axis=1)[maskpluvelx]
    FplusL[2][maskpluvelx] = np.roll(FL2, 1, axis=1)[maskpluvelx]
    FplusL[3][maskpluvelx] = np.roll(FL3, 1, axis=1)[maskpluvelx]


    # up, y direction
        #negative velocity
    FminU[0][maskminvely] = np.roll(FU0, -1, axis=0)[maskminvely]
    FminU[1][maskminvely] = np.roll(FU1, -1, axis=0)[maskminvely]
    FminU[2][maskminvely] = np.roll(FU2, -1, axis=0)[maskminvely]
    FminU[3][maskminvely] = np.roll(FU3, -1, axis=0)[maskminvely]
    
    FplusU[0][maskminvely] = FU0[maskminvely]
    FplusU[1][maskminvely] = FU1[maskminvely]
    FplusU[2][maskminvely] = FU2[maskminvely]
    FplusU[3][maskminvely] = FU3[maskminvely]

        #positive velocity
    FminU[0][maskpluvely] = FU0[maskpluvely]
    FminU[1][maskpluvely] = FU1[maskpluvely]
    FminU[2][maskpluvely] = FU2[maskpluvely]
    FminU[3][maskpluvely] = FU3[maskpluvely]

    FplusU[0][maskpluvely] = np.roll(FU0, 1, axis=0)[maskpluvely]
    FplusU[1][maskpluvely] = np.roll(FU1, 1, axis=0)[maskpluvely]
    FplusU[2][maskpluvely] = np.roll(FU2, 1, axis=0)[maskpluvely]
    FplusU[3][maskpluvely] = np.roll(FU3, 1, axis=0)[maskpluvely]

    # right, x direction
        # negative velocity
    FminR[0][maskminvelx] = np.roll(FR0, -1, axis=1)[maskminvelx]
    FminR[1][maskminvelx] = np.roll(FR1, -1, axis=1)[maskminvelx]
    FminR[2][maskminvelx] = np.roll(FR2, -1, axis=1)[maskminvelx]
    FminR[3][maskminvelx] = np.roll(FR3, -1, axis=1)[maskminvelx]

    FplusR[0][maskminvelx] = FR0[maskminvelx]
    FplusR[1][maskminvelx] = FR1[maskminvelx]
    FplusR[2][maskminvelx] = FR2[maskminvelx]
    FplusR[3][maskminvelx] = FR3[maskminvelx]

        # positive velocity
    FminR[0][maskpluvelx] = FR0[maskpluvelx]
    FminR[1][maskpluvelx] = FR1[maskpluvelx]
    FminR[2][maskpluvelx] = FR2[maskpluvelx]
    FminR[3][maskpluvelx] = FR3[maskpluvelx]

    FplusR[0][maskpluvelx] = np.roll(FR0, 1, axis=1)[maskpluvelx]
    FplusR[1][maskpluvelx] = np.roll(FR1, 1, axis=1)[maskpluvelx]
    FplusR[2][maskpluvelx] = np.roll(FR2, 1, axis=1)[maskpluvelx]
    FplusR[3][maskpluvelx] = np.roll(FR3, 1, axis=1)[maskpluvelx]
    return FplusL, FplusR, FminL, FminR, FplusD, FplusU, FminD, FminU




############################################################################################""
def getUs(vel, velLx, velRx, velUx, velDx, velLy, velRy, velUy, velDy, rhoL, rhoR, rhoU, rhoD, EL, ER, EU, ED, xcells, ycells):
    """Given the primitive variables, return the concerved variables U"""
    
    UminL = np.zeros((4, xcells, ycells))
    UminR = np.zeros((4, xcells, ycells))
    UplusL = np.zeros((4, xcells, ycells))
    UplusR = np.zeros((4, xcells, ycells))

    UminD = np.zeros((4, xcells, ycells))
    UminU = np.zeros((4, xcells, ycells))
    UplusD = np.zeros((4, xcells, ycells))
    UplusU = np.zeros((4, xcells, ycells))

    #masks to avoid loops:
    maskminvely = vel[1] < 0
    maskpluvely = vel[1] >= 0

    maskminvelx = vel[0] < 0
    maskpluvelx = vel[1] >= 0

    # Get all possible structures
    UD0 = rhoD
    UD1 = rhoD*velDx
    UD2 = rhoD*velDy
    UD3 = ED

    UR0 = rhoR
    UR1 = rhoR*velRx
    UR2 = rhoR*velRy
    UR3 = ER

    UL0 = rhoL
    UL1 = rhoL*velLx
    UL2 = rhoL*velLy
    UL3 = EL

    UU0 = rhoU
    UU1 = rhoU*velUx
    UU2 = rhoU*velUy
    UU3 = EU

    # down, y direction
        #negative velocity
    UminD[0][maskminvely] = np.roll(UD0, -1, axis=0)[maskminvely]
    UminD[1][maskminvely] = np.roll(UD1, -1, axis=0)[maskminvely]
    UminD[2][maskminvely] = np.roll(UD2, -1, axis=0)[maskminvely]
    UminD[3][maskminvely] = np.roll(UD3, -1, axis=0)[maskminvely]
    
    UplusD[0][maskminvely] = UD0[maskminvely]
    UplusD[1][maskminvely] = UD1[maskminvely]
    UplusD[2][maskminvely] = UD2[maskminvely]
    UplusD[3][maskminvely] = UD3[maskminvely]

        #positive velocity
    UminD[0][maskpluvely] = UD0[maskpluvely]
    UminD[1][maskpluvely] = UD1[maskpluvely]
    UminD[2][maskpluvely] = UD2[maskpluvely]
    UminD[3][maskpluvely] = UD3[maskpluvely]

    UplusD[0][maskpluvely] = np.roll(UD0, 1, axis=0)[maskpluvely]
    UplusD[1][maskpluvely] = np.roll(UD1, 1, axis=0)[maskpluvely]
    UplusD[2][maskpluvely] = np.roll(UD2, 1, axis=0)[maskpluvely]
    UplusD[3][maskpluvely] = np.roll(UD3, 1, axis=0)[maskpluvely]

    # left, x direction
        # negative velocity
    UminL[0][maskminvelx] = np.roll(UL0, -1, axis=1)[maskminvelx]
    UminL[1][maskminvelx] = np.roll(UL1, -1, axis=1)[maskminvelx]
    UminL[2][maskminvelx] = np.roll(UL2, -1, axis=1)[maskminvelx]
    UminL[3][maskminvelx] = np.roll(UL3, -1, axis=1)[maskminvelx]

    UplusL[0][maskminvelx] = UL0[maskminvelx]
    UplusL[1][maskminvelx] = UL1[maskminvelx]
    UplusL[2][maskminvelx] = UL2[maskminvelx]
    UplusL[3][maskminvelx] = UL3[maskminvelx]

        # positive velocity
    UminL[0][maskpluvelx] = UL0[maskpluvelx]
    UminL[1][maskpluvelx] = UL1[maskpluvelx]
    UminL[2][maskpluvelx] = UL2[maskpluvelx]
    UminL[3][maskpluvelx] = UL3[maskpluvelx]

    UplusL[0][maskpluvelx] = np.roll(UL0, 1, axis=1)[maskpluvelx]
    UplusL[1][maskpluvelx] = np.roll(UL1, 1, axis=1)[maskpluvelx]
    UplusL[2][maskpluvelx] = np.roll(UL2, 1, axis=1)[maskpluvelx]
    UplusL[3][maskpluvelx] = np.roll(UL3, 1, axis=1)[maskpluvelx]


    # up, y direction
        #negative velocity
    UminU[0][maskminvely] = np.roll(UU0, -1, axis=0)[maskminvely]
    UminU[1][maskminvely] = np.roll(UU1, -1, axis=0)[maskminvely]
    UminU[2][maskminvely] = np.roll(UU2, -1, axis=0)[maskminvely]
    UminU[3][maskminvely] = np.roll(UU3, -1, axis=0)[maskminvely]
    
    UplusU[0][maskminvely] = UU0[maskminvely]
    UplusU[1][maskminvely] = UU1[maskminvely]
    UplusU[2][maskminvely] = UU2[maskminvely]
    UplusU[3][maskminvely] = UU3[maskminvely]

        #positive velocity
    UminU[0][maskpluvely] = UU0[maskpluvely]
    UminU[1][maskpluvely] = UU1[maskpluvely]
    UminU[2][maskpluvely] = UU2[maskpluvely]
    UminU[3][maskpluvely] = UU3[maskpluvely]

    UplusU[0][maskpluvely] = np.roll(UU0, 1, axis=0)[maskpluvely]
    UplusU[1][maskpluvely] = np.roll(UU1, 1, axis=0)[maskpluvely]
    UplusU[2][maskpluvely] = np.roll(UU2, 1, axis=0)[maskpluvely]
    UplusU[3][maskpluvely] = np.roll(UU3, 1, axis=0)[maskpluvely]

    # right, x direction
        # negative velocity
    UminR[0][maskminvelx] = np.roll(UR0, -1, axis=1)[maskminvelx]
    UminR[1][maskminvelx] = np.roll(UR1, -1, axis=1)[maskminvelx]
    UminR[2][maskminvelx] = np.roll(UR2, -1, axis=1)[maskminvelx]
    UminR[3][maskminvelx] = np.roll(UR3, -1, axis=1)[maskminvelx]

    UplusR[0][maskminvelx] = UR0[maskminvelx]
    UplusR[1][maskminvelx] = UR1[maskminvelx]
    UplusR[2][maskminvelx] = UR2[maskminvelx]
    UplusR[3][maskminvelx] = UR3[maskminvelx]

        # positive velocity
    UminR[0][maskpluvelx] = UR0[maskpluvelx]
    UminR[1][maskpluvelx] = UR1[maskpluvelx]
    UminR[2][maskpluvelx] = UR2[maskpluvelx]
    UminR[3][maskpluvelx] = UR3[maskpluvelx]

    UplusR[0][maskpluvelx] = np.roll(UR0, 1, axis=1)[maskpluvelx]
    UplusR[1][maskpluvelx] = np.roll(UR1, 1, axis=1)[maskpluvelx]
    UplusR[2][maskpluvelx] = np.roll(UR2, 1, axis=1)[maskpluvelx]
    UplusR[3][maskpluvelx] = np.roll(UR3, 1, axis=1)[maskpluvelx]

    return UplusL, UplusR, UminL, UminR, UplusD, UplusU, UminD, UminU