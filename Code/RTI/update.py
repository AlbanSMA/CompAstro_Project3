import numpy as np
from primitives import getDensity, getVelocities, getEnergies
from UsandFs import getFluxes, getUs

def get_interface(rho, vel, Eint, phi, xcells, ycells, n_dim, dx, dy, adiab_ind):
   """Given the current density and velocity of the fluid, get the flux at the 
   interfaces: left and right in x, up and down in y."""

   # First the primitive variables
      # Density:
   rhoL, rhoR, rhoD, rhoU = getDensity(rho, n_dim, xcells, ycells, dx, dy)

      # Velocities
   velLx, velRx, velDx, velUx = getVelocities(vel[0], n_dim, xcells, ycells, dx, dy)
   velLy, velRy, velDy, velUy = getVelocities(vel[1], n_dim, xcells, ycells, dx, dy)

      #and magnitudes to calculate the energies:
   velL2 = velLx**2 + velLy**2
   velR2 = velRx**2 + velRy**2
   velD2 = velDx**2 + velDy**2
   velU2 = velUx**2 + velUy**2

      # Energy
   EintL, EintR, EintD, EintU = getEnergies(Eint, n_dim, xcells, ycells, dx, dy)
   EL = rhoL * (velL2/2 + Eint + phi)
   ER = rhoL * (velR2/2 + Eint + phi)
   ED = rhoL * (velD2/2 + Eint + phi)
   EU = rhoL * (velU2/2 + Eint + phi)

      # Pressure 
   PL = rhoL * EintL * (adiab_ind - 1)
   PD = rhoD * EintD * (adiab_ind - 1)
   PR = rhoR * EintR * (adiab_ind - 1)
   PU = rhoU * EintU * (adiab_ind - 1)

      # Sound speed from P, rho and the adiabatic index
   c_sL = np.sqrt(adiab_ind * (PL/rhoL))
   c_sD = np.sqrt(adiab_ind * (PD/rhoD))
   c_sR = np.sqrt(adiab_ind * (PR/rhoR))
   c_sU = np.sqrt(adiab_ind * (PU/rhoU))

   # Fluxes
   FplusL, FplusR, FminL, FminR, FplusD, FplusU, FminD, FminU = getFluxes(vel, velLx, velRx, velUx, velDx, velLy, velRy, velUy, velDy, rhoL, rhoR, rhoU, rhoD, EL, ER, EU, ED, PL, PR, PU, PD, xcells, ycells)

   # Wave speeds
   lamR = np.maximum(0, np.maximum(velRx + c_sR, velLx + c_sL))
   lamL = np.minimum(0, np.minimum(velRx - c_sR, velLx - c_sL))

   lamU = np.maximum(0, np.maximum(velUy + c_sU, velDy + c_sD))
   lamD = np.minimum(0, np.minimum(velUy - c_sU, velDy - c_sD))

   # And the Us:
   UplusL, UplusR, UminL, UminR, UplusD, UplusU, UminD, UminU = getUs(vel, velLx, velRx, velUx, velDx, velLy, velRy, velUy, velDy, rhoL, rhoR, rhoU, rhoD, EL, ER, EU, ED, xcells, ycells)

   # And finally, the actual summed fluxes
   lamdiffx = lamR - lamL
   lamdiffy = lamU - lamD
   lamdiffx = lamdiffx.clip(min = 1e-12)
   lamdiffy = lamdiffy.clip(min = 1e-12)

   Fminx = (lamR * FminL - lamL * FminR + lamR * lamL * (UminR - UminL))/(lamdiffx)
   Fminy = (lamU * FminD - lamD * FminU + lamU * lamD * (UminU - UminD))/(lamdiffy)

   Fplusx = (lamR * FplusL - lamL * FplusR + lamR * lamL * (UplusR - UplusL))/(lamdiffx)
   Fplusy = (lamU * FplusD - lamD * FplusU + lamU * lamD * (UplusU - UplusD))/(lamdiffy)
   
   return Fminx, Fminy, Fplusx, Fplusy

def get_newU(U, S, Fminx, Fminy, Fplusx, Fplusy, dt, dx, dy):
   """For a given previous U, S, fluxes in the y and x directions and interval in time,
   x and y, returns the new values for the concerved variables."""
   newU = (-((Fplusx - Fminx)/dx + (Fplusy - Fminy)/dy) + S)*dt + U
   return newU