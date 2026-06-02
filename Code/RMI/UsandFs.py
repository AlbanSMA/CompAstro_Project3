import numpy as np

def getFluxes(velLx, velRx, velUx, velDx, velLy, velRy, velUy, velDy, rhoL, rhoR, rhoU, rhoD, EL, ER, EU, ED, PL, PR, PU, PD, xcells, ycells):
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
    FU3 = (EU + PU) * velUy

# down, y direction
    FminD[0] = np.roll(FD0, 1, axis=1)
    FminD[1] = np.roll(FD1, 1, axis=1)
    FminD[2] = np.roll(FD2, 1, axis=1)
    FminD[3] = np.roll(FD3, 1, axis=1)
    
    FplusD[0] = FD0
    FplusD[1] = FD1
    FplusD[2] = FD2
    FplusD[3] = FD3

    # left, x direction
    FminL[0] = np.roll(FL0, 1, axis=0)
    FminL[1] = np.roll(FL1, 1, axis=0)
    FminL[2] = np.roll(FL2, 1, axis=0)
    FminL[3] = np.roll(FL3, 1, axis=0)

    FplusL[0] = FL0
    FplusL[1] = FL1
    FplusL[2] = FL2
    FplusL[3] = FL3

    # up, y direction
    FminU[0] = FU0
    FminU[1] = FU1
    FminU[2] = FU2
    FminU[3] = FU3
    
    FplusU[0] = np.roll(FU0, -1, axis=1)
    FplusU[1] = np.roll(FU1, -1, axis=1)
    FplusU[2] = np.roll(FU2, -1, axis=1)
    FplusU[3] = np.roll(FU3, -1, axis=1)

    # right, x direction
        # negative velocity
    FminR[0] = FR0
    FminR[1] = FR1
    FminR[2] = FR2
    FminR[3] = FR3

    FplusR[0] = np.roll(FR0, -1, axis=0)
    FplusR[1] = np.roll(FR1, -1, axis=0)
    FplusR[2] = np.roll(FR2, -1, axis=0)
    FplusR[3] = np.roll(FR3, -1, axis=0)
    return FplusL, FplusR, FminL, FminR, FplusD, FplusU, FminD, FminU




############################################################################################""
def getUs(velLx, velRx, velUx, velDx, velLy, velRy, velUy, velDy, rhoL, rhoR, rhoU, rhoD, EL, ER, EU, ED, xcells, ycells):
    """Given the primitive variables, return the concerved variables U"""
    
    UminL = np.zeros((4, xcells, ycells))
    UminR = np.zeros((4, xcells, ycells))
    UplusL = np.zeros((4, xcells, ycells))
    UplusR = np.zeros((4, xcells, ycells))

    UminD = np.zeros((4, xcells, ycells))
    UminU = np.zeros((4, xcells, ycells))
    UplusD = np.zeros((4, xcells, ycells))
    UplusU = np.zeros((4, xcells, ycells))

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
    # down from the minus boundary is in the m-1 cell
    UminD[0] = np.roll(UD0, 1, axis=1)
    UminD[1] = np.roll(UD1, 1, axis=1)
    UminD[2] = np.roll(UD2, 1, axis=1)
    UminD[3] = np.roll(UD3, 1, axis=1)

    UplusD[0] = UD0
    UplusD[1] = UD1
    UplusD[2] = UD2
    UplusD[3] = UD3

    # left, x direction
    # left from the minus boundary is in the m-1 cell
    UminL[0] = np.roll(UL0, 1, axis=0)
    UminL[1] = np.roll(UL1, 1, axis=0)
    UminL[2] = np.roll(UL2, 1, axis=0)
    UminL[3] = np.roll(UL3, 1, axis=0)

    UplusL[0] = UL0
    UplusL[1] = UL1
    UplusL[2] = UL2
    UplusL[3] = UL3

    # up, y direction
    # up from the plus boundary is in the m+1 cell
    UminU[0] = UU0
    UminU[1] = UU1
    UminU[2] = UU2
    UminU[3] = UU3
    
    UplusU[0] = np.roll(UU0, -1, axis=1)
    UplusU[1] = np.roll(UU1, -1, axis=1)
    UplusU[2] = np.roll(UU2, -1, axis=1)
    UplusU[3] = np.roll(UU3, -1, axis=1)

    # right, x direction
    # right from the plus boundary is in the m+1 cell
    UminR[0] = UR0
    UminR[1] = UR1
    UminR[2] = UR2
    UminR[3] = UR3

    UplusR[0] = np.roll(UR0, -1, axis=0)
    UplusR[1] = np.roll(UR1, -1, axis=0)
    UplusR[2] = np.roll(UR2, -1, axis=0)
    UplusR[3] = np.roll(UR3, -1, axis=0)
    return UplusL, UplusR, UminL, UminR, UplusD, UplusU, UminD, UminU