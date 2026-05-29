import numpy as np
from primitives import getDensity, getVelocities, getEnergies
from UsandFs import getFluxes, getUs

def get_interface(rho, vel, Eint, xcells, ycells, n_dim, dx, dy, adiab_ind):
   """Given the current density and velocity of the fluid, get the flux at the 
   interfaces: left and right in x, up and down in y."""

   # First the primitive variables
      # Density:
   rhoL, rhoR, rhoD, rhoU = getDensity(rho, n_dim, xcells, ycells, dx, dy)

      # Velocities
   velL, velR, velD, velU = getVelocities(vel, n_dim, xcells, ycells, dx, dy)

      # Internal energy
   EintL, EintR, EintD, EintU = getEnergies(Eint, n_dim, xcells, ycells, dx, dy)

      # Pressure from the equation of state
   #pressure looks a bit weird
   PL = rhoL * EintL * (adiab_ind - 1)
   PD = rhoD * EintD * (adiab_ind - 1)
   PR = rhoR * EintR * (adiab_ind - 1)
   PU = rhoU * EintU * (adiab_ind - 1)

      # Sound speed from P, rho and the adiabatic index
   c_sL = np.sqrt(adiab_ind * (PL/rhoL))
   c_sD = np.sqrt(adiab_ind * (PD/rhoD))
   c_sR = np.sqrt(adiab_ind * (PR/rhoR))
   c_sU = np.sqrt(adiab_ind * (PU/rhoU))

   print("rho", rhoL[49:52, 149:152], rhoD[49:52, 149:152])
   print("vel", velL[49:52, 149:152], velD[49:52, 149:152])
   print("Eint", EintL[49:52, 149:152], EintD[49:52, 149:152])
   print("P", PL[49:52, 149:152], PD[49:52, 149:152])
   print("c_s", c_sL[49:52, 149:152], c_sD[49:52, 149:152])


   # Fluxes
   FplusL, FplusR, FminL, FminR, FplusD, FplusU, FminD, FminU = getFluxes(vel, velL, velR, velU, velD, rhoL, rhoR, rhoU, rhoD, EintL, EintR, EintU, EintD, PL, PR, PU, PD, xcells, ycells)
            

   # Wave speeds
   lamR = np.maximum(0, velR + c_sR**2, velL + c_sL**2)
   lamL = np.minimum(0, velR - c_sR**2, velL - c_sL**2)

   lamU = np.maximum(0, velU + c_sU**2, velD + c_sD**2)
   lamD = np.minimum(0, velU - c_sU**2, velD - c_sD**2)

   # And the Us:
   UplusL, UplusR, UminL, UminR, UplusD, UplusU, UminD, UminU = getUs(vel, velL, velR, velU, velD, rhoL, rhoR, rhoU, rhoD, EintL, EintR, EintU, EintD, PL, PR, PU, PD, xcells, ycells)

   # And finally, the actual summed fluxes
   Fminx = (lamR * FminL - lamL * FminR + lamR * lamL * (UminR - UminL))/(lamR - lamL)
   Fminy = (lamU * FminD - lamD * FminU + lamU * lamD * (UminU - UminD))/(lamU - lamD)

   Fplusx = (lamR * FplusL - lamL * FplusR + lamR * lamL * (UplusR - UplusL))/(lamR - lamL)
   Fplusy = (lamU * FplusD - lamD * FplusU + lamU * lamD * (UplusU - UplusD))/(lamU - lamD)
   return Fminx, Fminy, Fplusx, Fplusy

def get_newU(U, S, Fminx, Fminy, Fplusx, Fplusy, dt, dx, dy):
   """For a given previous U, S, fluxes in the y and x directions and interval in time,
   x and y, returns the new values for the concerved variables."""
   newU = (-((Fplusx - Fminx)/dx - (Fplusy - Fminy)/dy) + S)*dt + U
   return newU