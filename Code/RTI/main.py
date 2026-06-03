import shutil
import os

from params import *
from init import *
from plot import *
from update import *

#Create paths
path = ""       #path to the folder to store the plots
imagepath = path+folder
files = os.listdir(path)
for i in files:
    if i == folder:
        shutil.rmtree(imagepath)
os.mkdir(imagepath)

#Initialise everything
time = 0
print("time:", time)

pos, g, phi, rho, P, E, Eint, c_s, vel, lstx, lsty, dx, dy = initialisation(n_dim, xdim, ydim, xcells, ycells, P0, adiab_ind)
U = ini_U(rho, vel, E, xcells, ycells)

    #including empty arrays for later
S = np.zeros((4, xcells, ycells))
Stemp = np.zeros((4, xcells, ycells))
new_S = np.zeros((4, xcells, ycells))

veltemp = np.zeros((n_dim, xcells, ycells))

#Plot the initialisation
fig, ax, img = ini_plot(rho, lstx, lsty, xcells, ycells, imagepath)

n = 1
while time < tend:
    #step dt and time
    dt = C * min(dx, dy)/(np.max(np.abs(vel+c_s))*n_dim)
    time = time+dt
    print("step", n, ", time:", time)
    
    #First loop
    Fminxtemp, Fminytemp, Fplusxtemp, Fplusytemp = get_interface(rho, vel, Eint, phi, xcells, ycells, n_dim, adiab_ind)
    Stemp[2,:,:] = rho*g

    new_Utemp = get_newU(U, Stemp, Fminxtemp, Fminytemp, Fplusxtemp, Fplusytemp, dt/2, dx, dy)
    new_Utemp = getBoundaries(new_Utemp)

    rhotemp = new_Utemp[0,:,:]
    velxtemp = new_Utemp[1,:,:]/rhotemp
    velytemp = new_Utemp[2,:,:]/rhotemp

    Etemp = new_Utemp[3,:,:]
    Einttemp = (Etemp/rhotemp) - ((velxtemp**2 + velytemp**2))/2 - phi

    veltemp[0] = velxtemp
    veltemp[1] = velytemp

    Utemp = new_Utemp

    #Final values with new flux after dt
    #Fluxes and S
    Fminx, Fminy, Fplusx, Fplusy = get_interface(rhotemp, veltemp, Einttemp, phi, xcells, ycells, n_dim, adiab_ind)
    S[2,:,:] = rhotemp*g

    #get the new values
    new_U = get_newU(U, new_S, Fminx, Fminy, Fplusx, Fplusy, dt, dx, dy)
    new_U = getBoundaries(new_U)

    rho = new_U[0,:,:]
    velx = new_U[1,:,:]/rho
    vely = new_U[2,:,:]/rho

    E = new_U[3,:,:]
    Eint = (E/rho) - ((velx**2 + vely**2))/2 - phi

    vel[0] = velx
    vel[1] = vely

    P = Eint * rho * (adiab_ind - 1)
    c_s = np.sqrt(adiab_ind * (P/rho))

    #Then plot
    if n%400 == 0:
        fig, ax = plotting(rho, fig, ax, img, time, n, imagepath)

    #Reinitialise
    n+=1
    U = new_U