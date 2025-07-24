#Tools to analyse coral model (created 20/6)

from dynasymV203 import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integ
from scipy import signal as signal
import sympy as sp


def simpleBifur(para, span, rhoDOC,rhoDIC,rhoDON,cons = []):
    y0 = [4, 40, 0.04, 0.12]
    tEnd = 4000
    paraList = np.linspace(span[0],span[1],200)

    lastList = []
    for paraValue in paraList:
        cD = makeCons([(para,paraValue)]+cons)
        sol = integ.solve_ivp(endo, y0=y0, t_span=[0,tEnd], args=(cD,rhoDOC,rhoDIC,rhoDON), dense_output=False, method="Radau", max_step = np.inf, rtol=1e-8, atol = 1e-8, events=[EtoHDiv,extinctE])
        pH, *dummy = makeFuncs(sol.t,sol.y,cD,rhoDOC,rhoDIC,rhoDON)
        if sol.status == 1:
            lastList.append([sol.y[0,-1], 0, sol.y[0,-1]/sol.y[1,-1]])
        else:
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


def plotTempBifur(rhoDOC,rhoDIC,rhoDON):
    cons =[("KN",0.2),("KE",0.2),("umax", 0.07),("mE",0.4),("mH",0.03)]
    paraList, lastList = simpleBifur("s",[0.5,3],rhoDOC,rhoDIC,rhoDON,cons=cons)
    x = np.flip(1/paraList) # np.linspace(0,len(paraList),len(paraList)) - len(paraList)*(1-0.5)/(1.5-0.5)

    fig, ax = plt.figure(), plt.subplot()
    ax.plot(x, np.flip(lastList[:,2]),".", color="gold", label = "$E/H$", ms=1.7, alpha=1)
    twin = ax.twinx()
    twin.plot(x, np.flip(lastList[:,0]), label = "$E$", color="C2", marker=".", ls="", ms=1.7, alpha=1)
    twin.plot(x, np.flip(lastList[:,1]), label = "$H$", color ="C0", marker=".", ls="", ms=1.7, alpha=1)
    
    ax.set_ylabel("E/H at equilibrium")
    twin.set_ylabel("$E$ or $H$ biomass (mol C/m$^2$)")
    ax.set_xlabel("Host relative energy demand")
    ax.legend(loc="center left")
    twin.legend(loc="center right")
    plt.show()


def plotmEBifur(rhoDOC,rhoDIC,rhoDON):
    cons =[("s",0.8),("KN",0.2),("KE",0.2),("umax", 0.07),("mH",0.03)]
    paraList, lastList = simpleBifur("mE",[0.1,2],rhoDOC,rhoDIC,rhoDON,cons=cons)
    x = paraList

    fig, ax = plt.figure(), plt.subplot()
    ax.plot(x, lastList[:,2],".", color="gold", label = "$E/H$", ms=1.7, alpha=1)
    twin = ax.twinx()
    twin.plot(x, lastList[:,0], label = "$E$", color="C2", marker=".", ls="", ms=1.7, alpha=1)
    twin.plot(x, lastList[:,1], label = "$H$", color ="C0", marker=".", ls="", ms=1.7, alpha=1)
    
    ax.set_ylabel("E/H at equilibrium")
    twin.set_ylabel("$E$ or $H$ biomass (mol C/m$^2$)")
    ax.set_xlabel("$m_E")
    ax.legend(loc="center left")
    twin.legend(loc="center right")
    plt.show()


def plotUmaxAndKE(rhoDOC,rhoDIC,rhoDON):
    cons =[("KN",0.2),("mE",0.4),("mH",0.03)]

    fig, axs = plt.subplots(nrows=1,ncols=2,figsize=(13,4))
    twin = (axs[0].twinx(), axs[1].twinx())
    for i in range(2):
        if i == 0:
            x, y = simpleBifur("umax",[0.01,0.12],rhoDOC,rhoDIC,rhoDON,cons=cons)
        else:
            x, y = simpleBifur("KE",[0.001,0.4],rhoDOC,rhoDIC,rhoDON,cons=cons)

        axs[i].plot(x, y[:,2] ,".", color="gold", label = "$E/H$", ms=1.7, alpha=1)
        twin[i].plot(x, y[:,0], label = "$E$", color="C2", marker=".", ls="", ms=1.7, alpha=1)
        twin[i].plot(x, y[:,1], label = "$H$", color ="C0", marker=".", ls="", ms=1.7, alpha=1)
    
    
        axs[i].legend(loc="center left")
        twin[i].legend(loc="center right")

    axs[0].set_ylabel("E/H at equilibrium")
    twin[1].set_ylabel("$E$ or $H$ biomass (mol C/m$^2$)")
    axs[0].set_xlabel("$u_{max}$")
    axs[1].set_xlabel("$K_E$")
    plt.show()


def plotFitnessVsE(H,QE,QH, cons=[]):
    N = 1000
    E = np.linspace(0,20,N)
    y = np.array([E, H*np.ones(N),QE*np.ones(N),QH*np.ones(N)])
    cD = makeCons(cons)

    pH, pE, mH, mE, *dummy = makeFuncs(np.zeros(N),y,cD,rhoDOC,rhoDIC,rhoDON)
    
    fig, ax = plt.figure(), plt.subplot()
    ax.plot(E,pH-mH, "C0", label = "Host")
    ax.plot(E,pE-mE, "C2", label = "Endosymbiont")
    



    ax.hlines(0,-1,21,"k",linestyles="dashed",alpha=0.2)
    ax.set_ylim([min(pH)-0.05,max(pH)+0.1])
    ax.set_title(f"$e_H=${round(cD["s"]*(1-cD["QHmin"]/QH),2)}, $e_E=${round((1-cD["QEmin"]/QE),2)}")
    ax.set_ylabel("Fitness")
    ax.set_xlabel("E")
    ax.legend()
    plt.show()


def solveSomeEqs():
    a,b,rho,p,H,K,y = sp.symbols("a b rho p H K y")

    x = (1-a)*(rho+y)
    sol = sp.solve( y-(1-b)*p*(x*H)/(x*H+K), y )
    print(sol)


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
    
    #plotTempBifur(rhoDOC,rhoDIC,rhoDON)
    #plotFitnessVsE(80,0.04,0.15)
    #plotUmaxAndKE(rhoDOC,rhoDIC,rhoDON)
    plotmEBifur(rhoDOC,rhoDIC,rhoDON)
