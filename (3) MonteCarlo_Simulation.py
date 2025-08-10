############################################################################################################################
##### BASIC IDEA ###########################################################################################################
############################################################################################################################

# Import adjusted Electron_temp file and run

# ---> Commented out "matplotlib.use('QT4Agg')" as this backend seems to be uncompatible with my version of matplotlib
# ---> Tried two other versions (Qt5Agg and TkAgg) but only TkAgg worked and as i don't know anything about backends i let python choose an appropriate one
# ---> What is a backend?
# ---> Component responsible for rendering and visualizing plots


from Electron_temp import *
import matplotlib
# matplotlib.use('QT4Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import exp

# Adjust starting energy from 10.0 to 20.0 according to task
T0 = 20.0
Tmin = 1.0 
numElectrons = 10

def getElectronTrajectory(el, electrons):
    # This function returns a curve x, y, z of one electron and adds newly created electrons to the list
    x = [el.x]
    y = [el.y]
    z = [el.z]
    while el.T > Tmin and (0 <= el.z < 10) and (-10 < el.x < 10) and (-10 < el.y < 10):
        s = rand_dist(lambda x: exp(-x), 0, 100, 1)
        el.propagate(s * el.T / 10)
        newElectron = Electron(el.x, el.y, el.z, el.dphi, el.dtheta, 0)
        el.scatter(newElectron)
        electrons.append(newElectron)
        x.append(el.x)
        y.append(el.y)
        z.append(el.z)
    return x, y, z

def main():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    ax.set_zlim([0, 10])

    electrons = [Electron(0, 0, 0, 0, 0, T0) for i in range(numElectrons)]

    for el in electrons:
        if el.T < Tmin or el.z < 0 or el.z > 10:
            continue
        x, y, z = getElectronTrajectory(el, electrons)
        ax.plot(x, y, z, 'b-')
    plt.savefig("C://Users//svens//OneDrive//Desktop//Python//2_Computational_Physics//Simulation_und_Fit_experimenteller_Daten//Elektronen.png")
    plt.show()

if __name__ == '__main__':
    main()