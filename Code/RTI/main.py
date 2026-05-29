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

pos, g, rho, P, Eint, c_s, vel, lstx, lsty, dx, dy = initialisation(n_dim, xdim, ydim, xcells, ycells, P0, adiab_ind)
U = ini_U(rho, vel, Eint, xcells, ycells, n_dim)
quit()

#Plot the initialisation
fig, ax, img = ini_plot(rho, lstx, lsty, xcells, ycells, imagepath)

n = 1
while time < tend:
    #step dt and time
    dt = C * min(dx, dy)/np.max(abs(vel+c_s)*n_dim)
    time = time+dt
    print("time:", time)
    
    #Fluxes
    Fminx, Fminy, Fplusx, Fplusy = get_interface(rho, vel, Eint, xcells, ycells, n_dim, dx, dy, adiab_ind)
    
    #S
    S = np.zeros((3, xcells, ycells))
    S[1,:,:] = rho*g

    #get the new values
    new_U = get_newU(U, S, Fminx, Fminy, Fplusx, Fplusy, dt, dx, dy)

    new_rho = new_U[:,0,:,:]
    new_vel = new_U[:,1,:,:]/new_rho
    new_Eint = new_U[:,2,:,:]
    new_P = new_Eint * new_rho * (adiab_ind - 1)
    new_cs = np.sqrt(adiab_ind * (new_P/new_rho))

    mask = (new_rho[0] == new_rho[1])
    notmask = (new_rho[0] != new_rho[1])
    if mask.all() == True:
        plotting(new_rho[0], fig, ax, img, n, imagepath)
    else:
        print("rho1", new_rho[0,notmask])
        print("rho2", new_rho[1,notmask])
        print("There must be a mistake somewhere")
        quit()

    #Reinitialise
    n+=1
    U = new_U
    rho = new_rho[0]
    vel = new_vel
    Eint = new_Eint[0]
    P = new_P
    c_s = new_cs
