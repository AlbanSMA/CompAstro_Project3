import shutil
import os

from params import *
from init import *
from plot import *
from update import *

#Create paths
path = "C:\\Users\\User\\Documents\\Documents\\University\\Master_SU\\Year_1\\IIB\\CompAstro\\Projects\\Project3\\Plots\\"
imagepath = path+folder
files = os.listdir(path)
for i in files:
    if i == folder:
        shutil.rmtree(imagepath)
os.mkdir(imagepath)

#Initialise everything
time = 0
print("time:", time)

pos, g, phi, rho, P, E, c_s, vel, lstx, lsty, dx, dy = initialisation(n_dim, xdim, ydim, xcells, ycells, P0, adiab_ind)
U = ini_U(rho, vel, E, xcells, ycells, n_dim)

#Plot the initialisation
fig, ax, img = ini_plot(rho, lstx, lsty, xcells, ycells, imagepath)

n = 1
while time < tend:
    #step dt and time
    dt = C * min(dx, dy)/(np.max(np.abs(vel+c_s))*n_dim)
    time = time+dt
    print(n, "time:", time)
    
    #Fluxes
    Fminx, Fminy, Fplusx, Fplusy = get_interface(rho, vel, E, phi, xcells, ycells, n_dim, dx, dy, adiab_ind)

    #S
    S = np.zeros((4, xcells, ycells))
    S[1,:,:] = rho*g

    #get the new values
    new_U = get_newU(U, S, Fminx, Fminy, Fplusx, Fplusy, dt, dx, dy)
    new_U[:,:,0] = new_U[:,:,1]
    new_U[:,:,-1] = new_U[:,:,-2]
    new_U[:,0,:] = new_U[:,1,:]
    new_U[:,-1,:] = new_U[:,-2,:]

    rho = new_U[0,:,:]
    velx = new_U[1,:,:]/rho
    vely = new_U[2,:,:]/rho

    E = new_U[3,:,:]
    Eint = (E/rho) - (velx**2 + vely**2)/2 - phi

    P = Eint * rho * (adiab_ind - 1)
    
    c_s = np.sqrt(adiab_ind * (P/rho))
    print(Eint[Eint==np.min(Eint)], np.argwhere(Eint==np.min(Eint)))
    print(P[P==np.min(P)], np.argwhere(P==np.min(P)))
    print(rho[rho==np.min(rho)], np.argwhere(rho==np.min(rho)))


    if n%4 == 0:
        plotting(rho, fig, ax, img, time, n, imagepath)

    #Reinitialise
    n+=1
    U = new_U
    vel[0] = velx
    vel[1] = vely