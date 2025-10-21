#Tools to analyse coral model (created 20/6)

from model import *

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd

import scipy.integrate as integ
import scipy.signal as signal
import scipy.stats as stats
import scipy.differentiate as diff
import scipy.linalg as la
import scipy.optimize as opt
import sympy as sp

##### Simulation and basic plotting ############################
def simSystem(y0,tSpan,cons=[],tEval=None):
    cD  = makeCons(cons)
    sol = integ.solve_ivp(endo, y0=y0, t_span=tSpan, t_eval=tEval, args=(cD,), dense_output=False, method="Radau", vectorized=True,
                          max_step = np.inf, rtol=1e-8, atol = 1e-8, events=[symbDeath])
    if sol.status == 1:
        return sol.t_events[0][0], sol.y_events[0][0], makeFuncs(sol.t_events[0][0],sol.y_events[0][0],cD), cD
    if sol.status == -1:
        return None,"Nope",None,None
    funcs = makeFuncs(sol.t,sol.y,cD)
    return sol.t, sol.y, np.array(funcs), cD


def makeDf(sol,funcs):
    return


def plotSim(t,y,funcs):
    """Ploting a given simulation in matplotlib"""
    H, E, N, C = y
    rE, rH, pE, muH, muE, uH, uE, rhoPhoto, rhoDIN, rhoDOC, rhoDON, eH, eE = funcs

    fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1)
    
    ax1.semilogy(t,H,"C0",label="H")
    ax1.semilogy(t,E,"C2",label="E")
    twin1 = ax1.twinx()
    twin1.plot(t,E/H,"C1", label = "$E/H$")
    
    twin2 = ax2.twinx()
    twin2.plot(t,muE,"C2", label=r"$\mu_{E}$")
    twin2.plot(t,pE,"C2", ls ="dashed", label=r"$p_{E}$")
    ax2.plot(t,muH,"C0", label=r"$\mu_{H}$")
    ax2.plot(t,rhoPhoto,"C3", ls ="dashed", label=r"$\rho_{photo}$")
    ax2.plot(t,rhoDOC,"k", ls="dashed", label=r"$\rho_{Food}$")


    ax1.set_ylabel(r"mol C /m$^2$")
    twin1.set_ylabel("E biomass/H iomass")
    ax2.set_ylabel(r"d$^{-1}$")
    twin2.set_ylabel(r"d$^{-1}$")
    ax2.set_xlabel("d")


    ax1.legend(loc="center left")
    twin1.legend(loc="lower right")
    ax2.legend(loc="upper right")
    twin2.legend(loc="lower right")


def plotCN(t,y,funcs,cD):
    """Ploting nutrien and carbon flows for a given simulation"""
    H, E, N, C = y
    rE, rH, pE, muH, muE, uH, uE, rhoPhoto, rhoDIN, rhoDOC, rhoDON, eH, eE  = funcs

    fig, (ax1,ax2,ax3) = plt.subplots(nrows=3, ncols=1)

    ax1.plot(t,N,"C4", label = "$N$")
    twin1 = ax1.twinx()
    twin1.plot(t,C,"k",label="C")

    ax2.plot(t,uE,"C2", ls ="dashed", label=r"$u_E$")
    ax2.plot(t,uE*E/H,"C2", label=r"$u_E\frac{E}{H}$")
    ax2.plot(t,rhoDIN,"C0", ls ="dashed", label=r"$\rho_{DIN}$")
    ax2.plot(t,uH,"C0", label=r"$u_H$")
    ax2.plot(t,cD["dC"]*(cD["CI"]-C),"C4", ls ="dashed", label=r"$\delta_N (N_I-N)$")
 
    ax3.plot(t, rH,"C0", label=r"$r_{H}$")
    ax3.plot(t, rE*E/H,"C0", ls ="dashed", label=r"$r_{E}\frac{E}{H}$")
    ax3.plot(t, pE*E/H,"C2", ls ="dashed", label=r"$p_{E}\frac{E}{H}$")
    ax3.plot(t, cD["dC"]*(cD["CI"]-C),"k", ls ="dashed", label=r"$\delta_C (C_I-C)$")

    ax1.legend(loc="center left")
    ax2.legend()
    twin1.legend(loc="center right")
    ax3.legend()

    ax1.set_ylabel("mol N/mol C")
    twin1.set_ylabel("mol CO$_2$/mol C")
    ax2.set_ylabel("mol N/mol C/d")
    ax3.set_ylabel("mol C/mol C/d")
    ax3.set_xlabel("days")


##### Symbolic function ####################################################
def checkFixedPoint(nLimH, nLimE, cons=[]):
    """Function uses symbolic computation to find fixed points for a given stat (H and/or E under nutrient limitation)"""
    paraValues = makeCons(cons)

    [dH,dE,dN,dC] = endoSymbolic(nLimH,nLimE)
    dfSubbed = [dH.subs(paraValues),dE.subs(paraValues),dN.subs(paraValues),dC.subs(paraValues)]
    fixedPoints = sp.solve(dfSubbed,[H, E, N, C])
    #print(fixedPoints)
    return fixedPoints


def checkSymbolicStab(yStar, cD, printEig = False):
    """Need some more work I guess"""
    funcs = makeFuncs(np.nan,yStar,cD)
    state= [funcs[-2]<1/(1+cD[s]),  funcs[-1]<1/(1+cD[s])]

    F = sp.Matrix(endoSymbolic(state[0],state[1])).subs(cD)
    J = F.jacobian([H,E,N,C])
    
    eig = J.subs([(H,yStar[0]), (E,yStar[1]), (N,yStar[2]), (C,yStar[3])]).eigenvals()
    stab, numDir = True, 0

    for val, mult in eig.items():
        if printEig:
            print("Eigenvalue:", sp.N(val))
        if sp.re(sp.N(val)) > 0:
            stab = False
            numDir += 1

    return stab#, numDir


def _tupleInList(t,lst,tol=1e-4):
    if not lst:  #check if empty
        return False
    newLst = np.array(lst)
    diffs = np.abs(newLst - np.array(t))
    return np.any(np.all(diffs <= tol, axis=1))

def makeSymbBifur(para,span,cons=[]): 
    pValues = np.linspace(span[0],span[1],50)
    fixList = []
    for p in pValues:
        cD = makeCons(cons+[(para,p)])
        fixedPoints = sorted(checkFixedPoint(0,0,cons=[(para,p)])) + sorted(checkFixedPoint(0,1,cons=[(para,p)]))
        curatedFps = []
        for fp in fixedPoints:
            if all(val>=0 for val in fp) and not _tupleInList(fp,curatedFps): 
                curatedFps.append(fp)
        for i in range(len(curatedFps)):
            fp = curatedFps[i]
            stab = checkSymbolicStab(fp, cD)
            fixList.append( np.concatenate([ [p], fp, [i], [stab]]) )
            print(f"{para}: {p}, H: {round(fp[0],20)}, E: {round(fp[1],20)}, N: {round(fp[2],20)}, C: {round(fp[3],20)}, fp: {i}, stable = {stab}")
    return np.array(fixList)



##### Bifurcation diagrams by numerically solving for the fixed point #####
def checkStab(yStar,cD):
    """Numerically checking stability of fixed point
    
    Input
    yStar: array-like fixed point
    cD: dict or dataframe of parametervalues
    
    Output
    Bool: True if stable, False otherwise"""
    
    res = diff.jacobian(lambda y: endo(0,y,cD), yStar, order=10, initial_step=0.1, maxiter=100, tolerances={"rtol":1e-12, "atol":1e-12} )
    eig, _ = la.eig(res.df, check_finite=False)
    #print(res.df)
    print(f"eigenvalues: {max([np.real(val) for val in eig])}")
    if all(val.real<0 for val in eig):
        return True
    else:
        return False


def numBifur(para,pValues,y0,cons=[]): 
    fixList = []
    for p in pValues:
        cD = makeCons(cons+[(para,p)])
        sol = opt.root(lambda y: endo(0,y,cD), y0, method="hybr", tol=1e-12, options={"xtol":1e-12,"maxfev":0,"eps":0.1})
        stab = checkStab(sol.x, cD)
        fixList.append( np.concatenate([ [p],y0,[stab] ]) )
        print(f"{para}: {p}, H: {round(sol.x[0],3)}, E: {round(sol.x[1],10)}, N: {round(sol.x[2],10)}, C: {round(sol.x[3],10)}, stable = {stab}")
        y0 = sol.x
    return np.array(fixList)


def makeNumBifur(para,span):
    fixedPoints = checkFixedPoint(0,1) + checkFixedPoint(0,0)+ checkFixedPoint(1,0) + checkFixedPoint(1,1)
    standardVal = makeCons()[para]
    step = (span[1]-span[0])/100
    mList, m = ["s", "o", "^", ">", "H"] + ["o"]*20, 0
    for i in range(len(fixedPoints)):
        fp = np.array(fixedPoints[i],dtype="float64")
        if True:#all(val>0 for val in fp):
            for j in range(2):
                pList = np.arange(standardVal,span[j],(-1)**(j+1)*step)
                fixList = numBifur(para,pList,fp)
                for k in range(0,len(fixList[:,0]),1):
                    if j == 1 and k == 0: continue
                    c = None if fixList[k,-1] else "red" 
                    plt.plot(fixList[k,0], fixList[k,1], color = "C0", mec = c, marker = mList[m], ms= 5.0, alpha=0.6, ls="", label = "$H^*$")
                    plt.plot(fixList[k,0], fixList[k,2], color = "C2", mec = c, marker = mList[m], ms= 5.0, alpha=0.6, ls="", label = "$E^*$")
            m += 1
    plt.xlabel(para)
    plt.ylabel("$H^*$, $E^*$ (mol C/m$^2$)")
    #plt.savefig("figs2/num_bifur_" + para + ".png")


def makeNumBifur2(para,span,cons=[]):                          
    fixedPoints = checkFixedPoint(0,1,cons) + checkFixedPoint(0,0,cons) #+ checkFixedPoint(1,0) + checkFixedPoint(1,1)
    standardVal = makeCons(cons)[para]
    step = (span[1]-span[0])/300

    bTable = np.empty((0,6))
    for i in range(len(fixedPoints)):
        fp = np.array(fixedPoints[i],dtype="float64")
        if all(val>0 for val in fp):  ##Excludes trivial fixedpoint...
            for j in range(2):
                pList = np.arange(standardVal,span[j],(-1)**(j+1)*step)
                fixList = numBifur(para,pList,fp,cons)
                if j==0:
                    bTable = np.r_[bTable, fixList[::-1]] 
                else: 
                    bTable = np.r_[bTable, fixList[1:,:]] 
    return bTable


##### Bifurcation diagrams by simulations #####################################
def simpleBifur(para, span, cons = []):    ## NB Modified simSystem before 
    y0 = [60, 1, 0.1, 0.1]                 ## Add y0 argument
    tEnd = 1000
    paraList = np.linspace(span[0],span[1],200)

    lastList = [[],[],[]]
    for paraValue in paraList:
        print(paraValue)
        newCons = cons + [(para,paraValue)]
        sol, funcs = simSystem(y0,[0,tEnd],newCons)
        lastList[0].append(sol.y[0,-1])
        lastList[1].append(sol.y[1,-1])
        lastList[2].append(funcs[7][-1])

        y0 = sol.y[:,-1]
        
    return [paraList] + lastList


def findOsc(y, tol = 1e-3):
    """Checks if vector oscilate at some period and returns mins and max of the oscillations

    Arguments:
    y: array like, vector of which oscillation is check (OBS: should be evenly spaced in timesteps)
    tol: float tolerence of solution 

    Returns:
    bool: False if convergence to fixed point, True otherwise
    list: Max and min values as lists
    """
    cv = stats.variation(y)  ## checking cv to see if no oscillations are occuring
    if cv<=tol:
        return [y[-1]], [y[-1]]
    
    maxIndex, _ = signal.find_peaks(y)
    if len(maxIndex) == 0:
        return [y[-1]], [y[-1]] 
    minIndex, _ = signal.find_peaks(-y)
    yMax, yMin = [], []
    for i in range(len(maxIndex)-1):
        yMax.append(y[maxIndex[-i]])
        if abs(y[maxIndex[-i]]-y[maxIndex[-i-1]])<=tol:
            break
    for j in range(len(minIndex)-1):
        yMin.append(y[minIndex[-j]])
        if abs(y[minIndex[-j]]-y[minIndex[-j-1]])<=tol:
            break

    return yMax, yMin
    

def bifur(para,span, cons = [], initVal = None):
    y0 = initVal or [60, 0.001, 0.01, 0.14]
    tEnd = 2000
    paraList = np.linspace(span[0],span[1],200)

    minMax = [[], [], [], []]
    for paraValue in paraList:
        print(paraValue)
        newCons = cons + [(para,paraValue)]
        try:
            t, y, funcs, cD = simSystem(y0,[0,tEnd],newCons,tEval=np.linspace(9*tEnd//10, tEnd, tEnd*100//10))
        except:
            continue
        
        if np.isscalar(y[0]):
            minMax[0] = minMax[0] + [paraValue]
            minMax[1] = minMax[1] + [y[0]]
            minMax[2] = minMax[2] + [y[1]]
            minMax[3] = minMax[3] + [funcs[7]]
            y0 = y
            continue


        HMax, HMin = findOsc(y[0,:])
        EMax, EMin = findOsc(y[1,:])
        photoMax, photoMin = findOsc(funcs[7])
        HLen, ELen = len(HMax)+len(HMin), len(EMax)+len(EMin)
        photoLen  = len(photoMax)+len(photoMin)

        minMax[0] = minMax[0] + [paraValue]*max(HLen,ELen,photoLen)
        minMax[1] = minMax[1] + HMax + HMin + max(0,ELen-HLen,photoLen-HLen)*[np.nan]
        minMax[2] = minMax[2] + EMax + EMin + max(0,HLen-ELen,photoLen-ELen)*[np.nan]
        minMax[3] = minMax[3] + photoMax + photoMin + max(0,ELen-photoLen,HLen-photoLen)*[np.nan]

        y0 = y[:,-1] + 1e-4

    return minMax


def saveBifur(para, Y):
    fig, tempAx = plt.subplots()
    p, H, E, rhoPhoto = Y
    tempAx.plot(p, rhoPhoto,".", color="C3", label = r"$\rho_{photo}$", ms=4.5, alpha=1)
    tempTwin = tempAx.twinx()
    tempTwin.plot(p, H, label = "$H$", color ="C0", marker=".", ls="", ms=3.5, alpha=1)
    tempTwin.plot(p, E, label = "$E$", color="C2", marker=".", ls="", ms=3.5, alpha=1)
    
    tempAx.set_xlabel(para)
    tempAx.set_ylabel(r"$\rho_{photo}$ at equilibrium")
    tempTwin.set_ylabel("$E$ and $H$ (mol C/m$^2$) at equilibrium")
    tempAx.legend(loc="center left")
    tempTwin.legend(loc="center right")
    plt.savefig("figs/bifur_" + para + ".png")
    plt.close()


def plotBifurOld(para,span,cons, initVal=None, bFunc = simpleBifur, ax = None, save = False):
    Y = np.array(bFunc(para,span,initVal=initVal,cons=cons))
    p, H, E, rhoPhoto = Y
    
    if ax==None:
        fig, ax = plt.subplots()
    if save:
        saveBifur(para,Y)

    ax.plot(p, rhoPhoto,".", color="C3", label = r"$\rho_{photo}$", ms=4.5, alpha=1)
    twin = ax.twinx()
    twin.plot(p, H, label = "$H$", color="C0", marker=".", ls="", ms=3.5, alpha=1)
    twin.plot(p, E, label = "$E$", color="C2", marker=".", ls="", ms=3.5, alpha=1)
    
    ax.set_xlabel(para)
    ax.legend(loc="center left")
    twin.legend(loc="center right")


###### Additional tools ########################################################
def checkCollision(fpList):  #omg a recursive function :0
    """Checks if a lsit of fixed points (given as tuples) contain any doubles 
    
    """
    if fpList == []:
        return False
    if _tupleInList(fpList[0], fpList[1:],tol=1e-1):
        return True
    return checkCollision(fpList[1:])

l = [ (1,2.0001), (0,0), (1,2), (1,1)]
print(checkCollision(l))



if False:# __name__ == "__main__":
    cD = makeCons([("CI",0.07)])
    for i in np.arange(-0.5,0.5,0.01):
        for j in np.arange(-0.5,0.5,0.01):
            sol = opt.root(lambda y: endo(0,y,cD), [125.556, 18.6306115977, i, j], method="hybr", tol=1e-12, options={"xtol":1e-12,"maxfev":0,"eps":0.1})
            if i==-0.5:
                starList = [sol.x]
            if all(np.linalg.norm(sol.x-fp)>1e-4 for fp in starList):
                starList = np.r_[starList, [sol.x]]
                
            
    for fp in starList:
        if fp[1]>5:
            print(fp)


    
    #makeNumBifur(CI, [0,0.15])
    #plt.show()

    