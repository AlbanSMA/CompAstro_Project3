import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm

def ini_plot(var, lstx, lsty, xcells, ycells, imagepath):
    """For a given map of rho and the path to save the image to, creates a plot
    and saves it."""

    # plot the data
    fig, ax = plt.subplots(1,1)

    img = ax.imshow(var, cmap="viridis")

    # set the location labels on the axis
    ax.set_yticks(np.linspace(0, xcells, 5))
    ax.set_xticks(np.linspace(0, ycells, 5))

    ax.set_xticklabels(np.linspace(lsty[0], lsty[-1], 5))
    ax.set_yticklabels(np.linspace(lstx[0], lstx[-1], 5))

    # finish the plot
    ax.set_ylabel("x")
    ax.set_xlabel("y")
    ax.set_title("time 0")

    #save
    fig.colorbar(img, ax=ax, orientation="horizontal")
    fig.savefig(imagepath+"\\step0.pdf", dpi = 300, bbox_inches = "tight")
    return fig, ax, img


def plotting(var, fig, ax, img, time, n, imagepath):
    """For a given updated rho at timestep n, updates plot ax and saves it to 
    imagepath"""

    img.set_data(var)
    ax.set_title(f"time {time:.5e}")

    fig.canvas.draw()
    fig.canvas.flush_events()

    fig.savefig(imagepath+f"\\step{n}.pdf", dpi=300, bbox_inches="tight")
    return fig, ax