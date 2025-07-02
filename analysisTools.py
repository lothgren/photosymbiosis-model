#Tools to analyse coral model (created 20/6)

from dynasymV202 import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integ


def simpleBifur(para, span):
    y0 = [4, 40, 0.04, 0.16]
    tEnd = 4000
    paraList = np.linspace(span[0],span[1],100)

    lastList = []
    for paraValue in paraList:
        cD = makeCons([(para,paraValue)])
        sol = integ.solve_ivp(endo, y0=y0, t_span=[0,tEnd], args=(cD,), dense_output=False, method="Radau", max_step = np.inf, rtol=1e-8, atol = 1e-8, events=[EtoHDiv,extinctE])
        lastList.append([sol.y[0,-1], sol.y[1,-1], sol.y[0,-1]/sol.y[1,-1]])
    
    return paraList, np.array(lastList)



def plotSimpleBifur():
    paraList, lastList = simpleBifur("umax",[0.005,0.035])
    plt.semilogy(paraList, lastList[:,0], label = "$E$", color="C2", marker=".", ls="")
    plt.semilogy(paraList, lastList[:,1], label = "$H$", color ="C0", marker=".", ls="")
    #plt.figure()
    #plt.plot(paraList, lastList[:,2],".", color="gold", label = "$E/H$")
    plt.ylabel("$E$ or $H$ biomass after 3 years (mol C/m$^2$)")
    plt.xlabel(r"$u_{\max}$")
    plt.legend()
    plt.show()



if __name__ == "__main__":
   paraList, lastList = simpleBifur("KE",[0.0001,1])
   plt.plot(paraList, lastList[:,2],".", color="gold", label = "$E/H$")
   plt.show()