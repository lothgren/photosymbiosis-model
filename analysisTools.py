#Tools to analyse coral model (created 20/6)

from dynasymV202 import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integ
from scipy import signal as signal


def simpleBifur(para, span, cons = []):
    y0 = [4, 40, 0.04, 0.16]
    tEnd = 4000
    paraList = np.linspace(span[0],span[1],100)

    lastList = []
    for paraValue in paraList:
        cD = makeCons([(para,paraValue)]+cons)
        sol = integ.solve_ivp(endo, y0=y0, t_span=[0,tEnd], args=(cD,), dense_output=False, method="Radau", max_step = np.inf, rtol=1e-8, atol = 1e-8, events=[EtoHDiv,extinctE])
        lastList.append([sol.y[0,-1], sol.y[1,-1], sol.y[0,-1]/sol.y[1,-1]])
    
    return paraList, np.array(lastList)


def bifur(para,span,cons = [], initVal = None):
    y0 = initVal or [4, 40, 0.04, 0.16]
    tEnd = 4000
    paraList = np.linspace(span[0],span[1],1000)

    minMax = []
    for paraValue in paraList:
        cD = makeCons([(para,paraValue)]+cons)
        sol = integ.solve_ivp(endo, y0=y0, t_span=[0,tEnd], t_eval=np.linspace(0,tEnd,tEnd*100), args=(cD,), dense_output=False, method="Radau", max_step = np.inf, rtol=1e-8, atol = 1e-8, events=[EtoHDiv,extinctE])
        
        minMax.append()


def plotSimpleBifur():
    paraList, lastList = simpleBifur("umax",[0.005,0.035])
    fig, ax = plt.figure(), plt.subplot()
    ax.plot(paraList, lastList[:,2],".", color="gold", label = "$E/H$")
    twin = ax.twinx()
    twin.semilogy(paraList, lastList[:,0], label = "$E$", color="C2", marker=".", ls="")
    twin.semilogy(paraList, lastList[:,1], label = "$H$", color ="C0", marker=".", ls="")
    
    plt.ylabel("$E$ or $H$ biomass after 3 years (mol C/m$^2$)")
    plt.xlabel(r"$u_{\max}$")
    plt.legend()
    plt.show()



if __name__ == "__main__":
   paraList, lastList = simpleBifur("umax",[0.001,1],cons=[("b",0.5)])
   plt.plot(paraList, lastList[:,2],".", color="gold", label = "$E/H$")
   plt.show()