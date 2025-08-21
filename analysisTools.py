#Tools to analyse coral model (created 20/6)

from dynasymV203CarbonPool import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integ
from scipy import signal as signal
import sympy as sp


def simSystem(y0,tSpan,cons,envFlows):
    cD  = makeCons(cons)
    sol = integ.solve_ivp(endo, y0=y0, t_span=tSpan, args=(cD,envFlows), dense_output=False, method="Radau", max_step = np.inf, rtol=1e-8, atol = 1e-8, events=[extinctE])
    funcs = makeFuncs(sol.t,sol.y,cD,envFlows)
    return sol, funcs


def makeDf(sol,funcs):
    return


def plotSim(y):
    return


def simpleBifur(para, span, envFlows,cons = []):
    y0 = [1, 60, 0.04, 0.12, 0.1]
    tEnd = 4000
    paraList = np.linspace(span[0],span[1],200)

    lastList = []
    for paraValue in paraList:
        newCons = cons + [(para,paraValue)]
        sol, funcs = simSystem(y0,[0,4000],newCons,envFlows)
        pH, *dummy = funcs
        if sol.status == 1:
            lastList.append([sol.y[0,-1]*sol.y[1,-1], 0, sol.y[0,-1]])
        else:
            lastList.append([sol.y[0,-1]*sol.y[1,-1], sol.y[1,-1], sol.y[0,-1]])
    
    return paraList, np.array(lastList)


def bifur(para,span,cons = [], initVal = None): ## Under construction
    y0 = initVal or [4, 40, 0.04, 0.16]
    tEnd = 4000
    paraList = np.linspace(span[0],span[1],1000)

    minMax = []
    for paraValue in paraList:
        cD = makeCons([(para,paraValue)]+cons)
        sol = integ.solve_ivp(endo, y0=y0, t_span=[0,tEnd], t_eval=np.linspace(0,tEnd,tEnd*100), args=(cD,), dense_output=False, method="Radau", max_step = np.inf, rtol=1e-8, atol = 1e-8, events=[EtoHDiv,extinctE])
        
        minMax.append()


def plotBifur(para,span,envFlows,cons, bFunc = simpleBifur):

    x, y = bFunc(para,span,envFlows,cons=cons)

    fig, ax = plt.figure(), plt.subplot()
    ax.plot(x, y[:,2],".", color="gold", label = "$E/H$", ms=2, alpha=1)
    twin = ax.twinx()
    twin.plot(x, y[:,0], label = "$E$", color="C2", marker=".", ls="", ms=2, alpha=1)
    twin.plot(x, y[:,1], label = "$H$", color ="C0", marker=".", ls="", ms=2, alpha=1)
    
    ax.set_xlabel(para)
    ax.set_ylabel("E/H at equilibrium")
    twin.set_ylabel("$E$ and $H$ (mol C/m$^2$) at equilibrium")
    ax.legend(loc="center left")
    twin.legend(loc="center right")


def plotFitnessVsE(H,QE,QH, cons=[]):
    N = 1000
    E = np.linspace(0,20,N)
    cD = makeCons(cons)

    fig, ax = plt.figure(), plt.subplot()
    for i in range(len(H)):
        y = np.array([E, H[i]*np.ones(N),QE*np.ones(N),QH*np.ones(N)])
        pH, pE, mH, mE, *dummy = makeFuncs(np.zeros(N),y,cD,envFlows)

        ax.plot(E,pH-mH, "C0", dashes=[1+i,i], alpha=1-2*i/10)
        ax.plot(E,pE-mE, "C2", dashes=[1+i,i], alpha=1-2*i/10)

    plt.arrow(8,0.02,4.7-8,0,width=0.002,length_includes_head=True, head_length=1,color="k")
    plt.text(4.7,0.025,"E fit. decreases")

    ax.hlines(0,-1,21,"k",linestyles="dashed",alpha=0.2)
    ax.set_ylim([min(pH)-0.05,max(pH)+0.1])
    ax.set_title(f"$e_H=${round(cD["s"]*(1-cD["QHmin"]/QH),2)}, $e_E=${round((1-cD["QEmin"]/QE),2)}")
    ax.set_ylabel("Fitness")
    ax.set_xlabel("E")
    ax.legend(["H fitness", "E fitness"])
    plt.show()


def solveSomeEqs():
    a,b,rho,p,H,K,y = sp.symbols("a b rho p H K y")

    x = (1-a)*(rho+y)
    sol = sp.solve( y-(1-b)*p*(x*H)/(x*H+K), y )
    print(sol)


if __name__ == "__main__":
    def rhoDOC(t,y,cD):
        E, H, QE, QH = y[0], y[1], y[2], y[3]
        return 0.03 *2* (1-H/166)
    def rhoDIC(t,y,cD):
        E, H, QE, QH = y[0], y[1], y[2], y[3]
        return cD["mH"]*0.0
    def rhoDON(t,y,cD):
        E, H, QE, QH = y[0], y[1], y[2], y[3]
        return rhoDOC(t,y,cD) * 0.15 + cD["mH"]*0.0
    cons = []
    plotBifur("umax", [0.01,1.7], [rhoDOC,rhoDON], cons)
    plt.show()