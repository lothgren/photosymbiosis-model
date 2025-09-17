#Tools to analyse coral model (created 20/6)

from model import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integ
from scipy import signal as signal
import sympy as sp


def symbDeath(t,y,cD,envFlows):
    return y[1]-1e-10
symbDeath.terminal = True


def simSystem(y0,tSpan,cons,envFlows):
    cD  = makeCons(cons)
    sol = integ.solve_ivp(endo, y0=y0, t_span=tSpan, args=(cD,envFlows), dense_output=False, method="Radau", max_step = np.inf, rtol=1e-8, atol = 1e-8, events=[symbDeath])
    funcs = makeFuncs(sol.t,sol.y,cD,envFlows)
    return sol, funcs


def makeDf(sol,funcs):
    return


def plotSim(t,y):
    """Ploting a simulation in matplotlib"""
    return


def simpleBifur(para, span, envFlows, cons = []):
    y0 = [60, 1, 0.04, 0.12, 0.1]
    tEnd = 4000
    paraList = np.linspace(span[0],span[1],100)

    lastList = []
    for paraValue in paraList:
        newCons = cons + [(para,paraValue)]
        sol, funcs = simSystem(y0,[0,4000],newCons,envFlows)
        lastList.append([sol.y[1,-1]/sol.y[0,-1], sol.y[0,-1], sol.y[1,-1]])
        
    return paraList, np.array(lastList)


def findOsc(y, tol = 1e-3):
    """Checks if vector oscilate at some period and returns mins and max of the oscillations

    Arguments:
    y: array like, vector of which oscillation is check (OBS: should be evenly spaced in timesteps)
    tol: float tolerence of solution 

    Returns:
    bool: False if convergence to fixed point, True otherwise
    list: Max and min values as lists
    """
    cv = np.std(y)/np.mean(y)  ## checking cv to see if no oscillations are occuring
    if cv<=tol:
        return [y[-1]], [y[-1]]
    
    maxIndex, _ = signal.find_peaks(y)
    minIndex, _ = signal.find_peaks(-y)
    yMax, yMin = [], []
    for i in range(len(maxIndex)-1):
        yMax.append(y[maxIndex[-i]])
        if abs(y[maxIndex[-i]]-y[maxIndex[-i-1]])<=tol:
            break
    for j in range(len(minIndex)-1):
        yMin.append(y[minIndex[-i]])
        if abs(y[minIndex[-i]]-y[minIndex[-i-1]])<=tol:
            break

    return yMax, yMin
    


def bifur(para,span, envFlows, cons = [], initVal = None): ## Under construction
    y0 = initVal or [60, 1, 0.04, 0.12, 0.1]
    tEnd = 4000
    paraList = np.linspace(span[0],span[1],200)

    minMax = [[], [], []]
    for paraValue in paraList:
        cD = makeCons([(para,paraValue)]+cons)
        sol = integ.solve_ivp(endo, y0=y0, t_span=[0,tEnd], t_eval=np.linspace(7*tEnd//10,tEnd,3*tEnd*10), args=(cD,envFlows), 
                              dense_output=False, vectorized=True, method="Radau", max_step = np.inf, rtol=1e-8, atol = 1e-8, events=[])
        
        HMax, HMin = findOsc(sol.y[0,:])
        EMax, EMin = findOsc(sol.y[1,:])
        HLen, ELen = len(HMax)+len(HMin), len(EMax)+len(EMin)
        if HLen >= ELen:
            minMax[0] = minMax[0] + [paraValue]*HLen
            minMax[1] = minMax[1] + HMax + HMin
            minMax[2] = minMax[2] + EMax + EMin + (HLen-ELen)*[None]
        else:
            minMax[0] = minMax[0] + [paraValue]*ELen
            minMax[1] = minMax[1] + HMax + HMin + (ELen-HLen)*[None]
            minMax[2] = minMax[2] + EMax + EMin 

    return minMax
        


def plotBifur(para,span,envFlows,cons, bFunc = simpleBifur):

    x, y = bFunc(para,span,envFlows,cons=cons)
    
    fig, ax = plt.figure(), plt.subplot()
    ax.plot(x, y[:,0],".", color="gold", label = "$E/H$", ms=2, alpha=1)
    twin = ax.twinx()
    twin.plot(x, y[:,2], label = "$E$", color="C2", marker=".", ls="", ms=2, alpha=1)
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



if __name__ == "__main__":
    def rhoDOC(t,y,cD):
        H, E, QE, QH, C = y
        return 0.03 *3* (1-H/166)
    def rhoDON(t,y,cD):
        H, E, QE, QH, C = y
        return rhoDOC(t,y,cD) * 0.15 + cD["mH"]*QH*0.9
    
    cons = [("s",1), ("mH",0.03),("mE",0.3),("KN",0.005),("umax",0.02)]

    plotBifur("s", [0.25,2], [rhoDOC,rhoDON], cons)
    plt.show()
