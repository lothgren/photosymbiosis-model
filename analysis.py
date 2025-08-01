#Doing all the simulations, last updated 30/7

from dynasymV203 import *
from analysisTools import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integ
from scipy import signal as signal
import sympy as sp




def plotUmax(rhoDOC,rhoDIC,rhoDON, cons): 
    fig, ax = plt.figure(), plt.subplot()
    twin = ax.twinx()
    x, y = simpleBifur("umax",[0.01,0.12],rhoDOC,rhoDIC,rhoDON,cons=cons)
    
    ax.plot(x, y[:,2] ,".", color="gold", label = "$E/H$", ms=1.7, alpha=1)
    twin.plot(x, y[:,0], label = "$E$", color="C2", marker=".", ls="", ms=1.7, alpha=1)
    twin.plot(x, y[:,1], label = "$H$", color ="C0", marker=".", ls="", ms=1.7, alpha=1)

    ax.legend(loc="center left")
    twin.legend(loc="center right")

    ax.set_ylabel("E/H at equilibrium")
    twin.set_ylabel("$E$ or $H$ biomass (mol C/m$^2$)")
    ax.set_xlabel(r"$u_{\max}$")
    

def plotKE(rhoDOC,rhoDIC,rhoDON,cons):
    fig, ax = plt.figure(), plt.subplot()
    twin = ax.twinx()
    x, y = simpleBifur("KE",[0.001,0.4],rhoDOC,rhoDIC,rhoDON,cons=cons)

    ax.plot(x, y[:,2] ,".", color="gold", label = "$E/H$", ms=1.7, alpha=1)
    twin.plot(x, y[:,0], label = "$E$", color="C2", marker=".", ls="", ms=1.7, alpha=1)
    twin.plot(x, y[:,1], label = "$H$", color ="C0", marker=".", ls="", ms=1.7, alpha=1)


    ax.legend(loc="center left")
    twin.legend(loc="center right")

    ax.set_ylabel("E/H at equilibrium")
    twin.set_ylabel("$E$ or $H$ biomass (mol C/m$^2$)")
    ax.set_xlabel("$K_E$")


def plotTempBifur(rhoDOC,rhoDIC,rhoDON,cons):
    paraList, y = simpleBifur("s",[0.5,3],rhoDOC,rhoDIC,rhoDON,cons=cons)
    x = np.flip(1/paraList) # np.linspace(0,len(paraList),len(paraList)) - len(paraList)*(1-0.5)/(1.5-0.5)

    fig, ax = plt.figure(), plt.subplot()
    ax.plot(x, np.flip(y[:,2]),".", color="gold", label = "$E/H$", ms=1.7, alpha=1)
    twin = ax.twinx()
    twin.plot(x, np.flip(y[:,0]), label = "$E$", color="C2", marker=".", ls="", ms=1.7, alpha=1)
    twin.plot(x, np.flip(y[:,1]), label = "$H$", color ="C0", marker=".", ls="", ms=1.7, alpha=1)
    
    ax.set_ylabel("E/H at equilibrium")
    twin.set_ylabel("$E$ or $H$ biomass (mol C/m$^2$)")
    ax.set_xlabel("Host relative energy demand")
    ax.legend(loc="center left")
    twin.legend(loc="center right")


if __name__ == "__main__":

    def rhoDOC(t,y,cD):
        E, H, QE, QH = y[0], y[1], y[2], y[3]
        return 0.03 *4* (1-H/166)
    def rhoDIC(t,y,cD):
        E, H, QE, QH = y[0], y[1], y[2], y[3]
        return cD["mH"]*0.0
    def rhoDON(t,y,cD):
        E, H, QE, QH = y[0], y[1], y[2], y[3]
        return rhoDOC(t,y,cD) * 0.15 + cD["mH"]*0.0
    
    cons =[("umax",0.07),("KN",0.2),("KE",0.2),("mE",0.4),("mH",0.03)]


    #plotUmax(rhoDOC,rhoDIC,rhoDON,cons)
    #plotKE(rhoDOC,rhoDIC,rhoDON,cons)
    plotTempBifur(rhoDOC,rhoDIC,rhoDON,cons)
    plt.show()