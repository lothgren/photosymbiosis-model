#Tools to analyse coral model (created 20/6)

from model import *

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd

import jax

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
    ax2.plot(t,cD[delC]*(cD[NI]-N),"C4", ls ="dashed", label=r"$\delta_N (N_I-N)$")
 
    ax3.plot(t, rH,"C0", label=r"$r_{H}$")
    ax3.plot(t, rE*E/H,"C0", ls ="dashed", label=r"$r_{E}\frac{E}{H}$")
    ax3.plot(t, pE*E/H,"C2", ls ="dashed", label=r"$p_{E}\frac{E}{H}$")
    ax3.plot(t, cD[delC]*(cD[CI]-C),"k", ls ="dashed", label=r"$\delta_C (C_I-C)$")

    ax1.legend(loc="center left")
    ax2.legend()
    twin1.legend(loc="center right")
    ax3.legend()

    ax1.set_ylabel("mol N/mol C")
    twin1.set_ylabel("mol CO$_2$/mol C")
    ax2.set_ylabel("mol N/mol C/d")
    ax3.set_ylabel("mol C/mol C/d")
    ax3.set_xlabel("days")


def plot_sim_2(t,y,funcs,cD):
    """Ploting a given simulation in matplotlib"""
    H, E, N, C = y
    rE, rH, pE, muH, muE, uH, uE, rhoPhoto, rhoDIN, rhoDOC, rhoDON, eH, eE = funcs

    fig1, axs = plt.subplots(2,1)
    ax1, ax2 = axs[0],axs[1]
    ax1.semilogy(t,H,"C0",label="H")
    ax1.semilogy(t,E,"C2",label="E")
    twin1 = ax1.twinx()
    twin1.plot(t,E/H,"C1", label = "$E/H$")
    
    ax1.legend(loc="center left")
    twin1.legend(loc="center right")
    ax1.set_ylabel(r"mol C /m$^2$")
    twin1.set_ylabel("E biomass/H iomass")
    ax1.set_xlabel("$d$")


    ax2.plot(t,N,"C4", label = "$N$")
    twin2 = ax2.twinx()
    twin2.plot(t,C,"k",label="C")
    
    ax2.legend(loc="center left")
    twin2.legend(loc="center right")
    ax2.set_ylabel("mol N/mol C")
    twin2.set_ylabel("mol CO$_2$/mol C")
    ax2.set_xlabel("$d$")


    fig3, ax3 = plt.subplots(1,1)
    twin3 = ax3.twinx()
    twin3.plot(t,muE,"C2", label=r"$\mu_{E}$")
    twin3.plot(t,pE,"C2", ls ="dashed", label=r"$p_{E}$")
    ax3.plot(t,muH,"C0", label=r"$\mu_{H}$")
    ax3.plot(t,rhoPhoto,"C3", ls ="dashed", label=r"$\rho_{photo}$")
    ax3.plot(t,rhoDOC,"k", ls="dashed", label=r"$\rho_{Food}$")

    ax3.legend(loc="upper right")
    twin3.legend(loc="lower right")
    ax3.set_ylabel("$d^{-1}$")
    ax3.set_xlabel("d")


    fig4, ax4 = plt.subplots(1,1)
    ax4.plot(t,uE*E/H,"C2", label=r"$u_E\frac{E}{H}$")
    ax4.plot(t,uH - rhoDIN + (cD[mH]*cD[QH]+cD[mE]*cD[QE]*E/H),"C0", label="H net N-uptake")
    ax4.plot(t,0*E+(cD[mH]*cD[QH]+cD[mE]*cD[QE]*E/H), "C0", ls ="dashed", label=r"$m_HQ_H+m_EQ_E\frac{E}{H}$")
    ax4.plot(t,-cD[delN]*(cD[NI]-N),"C4", ls ="dashed", label=r"$\delta_N (N-N_I)$")

    ax4.legend()
    ax4.set_xlabel("d")
    ax4.set_ylabel("mol N/mol C/d")

    return ax1,ax2,ax3,ax4


##### Symbolic function ####################################################
def checkFixedPoint(nLimH, nLimE, cD):
    """Function uses symbolic computation to find fixed points for a given stat (H and/or E under nutrient limitation)"""

    [dH,dE,dN,dC] = endoSymbolic(nLimH,nLimE)
    dfSubbed = [dH.subs(cD),dE.subs(cD),dN.subs(cD),dC.subs(cD)]
    fixedPoints = sp.solve(dfSubbed,[H, E, N, C])
    #print(fixedPoints)
    return fixedPoints



def checkSymbolicStab(yStar, cD):
    """Check stability of fixed point by symbolically calculation jacobian and eigenvalues
    
    Arguments:
    yStar: array-like, giving the fixed point of the system
    cD: dict or dataframe, giving the parameter values of the system

    Returns
    True if all eigenvalues have negative real part, False otherwise
    """
    funcs = makeFuncs(np.nan,yStar,cD)
    state= [funcs[-2]<1/(1+cD[s]),  funcs[-1]<1/(1+cD[s])]

    F = sp.Matrix(endoSymbolic(state[0],state[1])).subs(cD)
    J = F.jacobian([H,E,N,C])
    
    eig = J.subs([(H,yStar[0]), (E,yStar[1]), (N,yStar[2]), (C,yStar[3])]).eigenvals()
    stab, numDir = True, 0

    for val, mult in eig.items():
        if sp.re(sp.N(val)) > 0:
            stab = False
            numDir += 1

    return stab#, numDir


def checkIllegal(state,yStar,cD):
    rE, rH, pE, muH, muE, uH, uE, rhoPhoto, rhoDIN, rhoDOC, rhoDON, eH, eE = makeFuncs(0,yStar,cD)
    currState = [bool(eH<(1/(cD[s]+1))) , bool(eE<(1/(cD[s]+1)))]
    return currState == state


def _key(n):
    return sum(n)


def assign_fp_number(fp,state,i): # There has to be a prettier way to do this
    if fp[1]<1e-2:
            return 0
    if state == [0,0]:
        return i + 1
    else:
        return i + 2 


def _curate_fps(fixed_points,state,cD):
    """Takes a list of fixed points symbolicly calculated for a specific state and returns all feasible fixed points sorted from largest to smallest.
    
    Args:
        fixed_points (array-like):  List of fps.
        state (list): 2D array giving the state of the calculated fixed points.
        cD (dict or dateframe): Parameter values used when calculating fixed points.

    Returns:
        list: Curated and sorted list of fixed points.
    """
    curated_fps = []
    for fp in fixed_points:
        if all(val>=0 for val in fp) and checkIllegal(state,fp,cD): 
            curated_fps.append(fp)
    return sorted(curated_fps,key=_key,reverse=True)


def find_all_fps(cons=[], ignore_H_1 = True):
    """Symbilicly finds all feasible fixed point of the function
    
    Args:"""
    state_list = [(0,0), (0,1)] if ignore_H_1 else [(0,0), (0,1), (1,1), (1,0)]
    cD = makeCons(cons)

    fps = []
    for i, j in state_list:
        fps = fps + _curate_fps(checkFixedPoint(i,j,cD), [i,j], cD)

    for i in range(len(fps)):
        fp = fps[i]
        if fp[1] == 0.0:
            fps.remove(fp)
            fps.insert(0,fp)

    return fps


def makeSymbBifur(para,span,state=[0,1],cons=[]): 
    pValues = np.linspace(span[0],span[1],45)
    fixList = []
    for p in pValues:
        cD = makeCons(cons+[(para,p)])
        fixedPoints = sorted(checkFixedPoint(state[0],state[1],cD),key=_key,reverse=True)
        curated_fps = _curate_fps(fixedPoints,state,cD)
        for i in range(len(curated_fps)):
            fp = curated_fps[i]
            num = assign_fp_number(fp,state,i)
            stab = checkSymbolicStab(fp, cD)
            fixList.append( np.concatenate([ [p], fp, [num], [stab]]) )
            print(f"{para}: {p}, H: {round(fp[0],20)}, E: {round(fp[1],20)}, N: {round(fp[2],20)}, C: {round(fp[3],20)}, fp: {num}, stable = {stab}")
    return np.array(fixList)


##### Bifurcation diagrams by using automatic differentiation #############
def make_jac(f):
    return jax.jacfwd(f)


def check_stab(yStar,cD):
    """Checking stability of fixed point using automatic differentiation
    
    Input
    yStar: array-like fixed point
    cD: dict or dataframe of parametervalues
    
    Output
    Bool: True if stable, False otherwise"""
    
    J = make_jac(lambda y: endo(0,y,cD,minFunc=jax.numpy.minimum))
    J_subbed = jax.numpy.array(J(jax.numpy.array(yStar)))
    eig, _ = jax.numpy.linalg.eig(J_subbed)
    if all(val.real<0 for val in eig):
        return True
    else:
        return False


def aut_bifur(para,p_values,y0,fp_num=None,cons=[]):
    fixList = []
    for p in p_values:
        cD = makeCons(cons+[(para,p)])
        f = lambda y: endo(0,y,cD,minFunc=minApprox)
        J = make_jac(f)
        sol = opt.root(f, y0, method="hybr", tol=1e-10, options={"xtol":1e-10,"maxfev":0,"eps":0.1})
        if sol.success and all(val>=0 for val in sol.x):
            fixList.append(np.concatenate( [[p], sol.x, [fp_num] ,[check_stab(sol.x,cD)]]))
        else:
            fixList.append( [p] + [np.nan]*len(sol.x) + [fp_num] + [False])
        y0 = sol.x
    return fixList


def make_aut_bifur(para,span,cons=[]):
    cD = makeCons(cons)
    standardVal = cD[para]
    step = (span[1]-span[0])/200

    curatedFps = []
    for state in [[0,0], [0,1] ]:     #, [1,1], [1,0]]:
        fixedPoints = sorted(checkFixedPoint(state[0],state[1],cD),key=_key,reverse=True)
        for fp in fixedPoints:
            if all(val>=0 for val in fp) and checkIllegal(state,fp,cD): 
                curatedFps.append(np.array(fp,dtype="float64"))
    curatedFps = sorted(curatedFps,key=_key)

    fix_list = []
    for i in range(len(curatedFps)):
        fp = curatedFps[i]
        for j in range(2):
            pList = np.arange(standardVal,span[j],(-1)**(j+1)*step)
            fix_list = fix_list + aut_bifur(para,pList,fp,fp_num=i,cons=cons)
    
    return np.array(fix_list)


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
        sol = opt.root(lambda y: endo(0,y,cD,minFunc=minApprox), y0, method="hybr", tol=1e-12, options={"xtol":1e-12,"maxfev":0,"eps":0.1})
        if sol.success and all(val>=0 for val in sol.x):
            stab = checkStab(sol.x, cD)
            fixList.append( np.concatenate([ [p],y0,[stab] ]) )
            print(f"{para}: {p}, H: {round(sol.x[0],3)}, E: {round(sol.x[1],10)}, N: {round(sol.x[2],10)}, C: {round(sol.x[3],10)}, stable = {stab}")
        y0 = sol.x
    return np.array(fixList)


def makeNumBifur(para,span,cons=[]):
    cD = makeCons(cons)
    fixedPoints = checkFixedPoint(0,1,cD) + checkFixedPoint(0,0,cD) # checkFixedPoint(1,0,cD) + checkFixedPoint(1,1,cD)
    standardVal = cD[para]
    step = (span[1]-span[0])/100
    mList, m = ["s", "o", "^", ">", "H"] + ["o"]*20, 0
    for i in range(len(fixedPoints)):
        fp = np.array(fixedPoints[i],dtype="float64")
        if all(val>=0 for val in fp):
            for j in range(2):
                pList = np.arange(standardVal,span[j],(-1)**(j+1)*step)
                fixList = numBifur(para,pList,fp,cons=cons)
                for k in range(0,len(fixList[:,0]),1):
                    if j == 1 and k == 0: continue
                    c = None if fixList[k,-1] else "red" 
                    plt.plot(fixList[k,0], fixList[k,1], color = "C0", mec = c, marker = mList[m], ms= 5.0, alpha=0.6, ls="", label = "$H^*$")
                    plt.plot(fixList[k,0], fixList[k,2], color = "C2", mec = c, marker = mList[m], ms= 5.0, alpha=0.6, ls="", label = "$E^*$")
            m += 1
    plt.xlabel(para)
    plt.ylabel("$H^*$, $E^*$ (mol C/m$^2$)")
    #plt.savefig("figs2/num_bifur_" + para + ".png")


def makeNumBifur2(para,span,fixedPoints,cons=[]):  ## Obs doesn't check if fixed points are viable!                        
    standardVal = makeCons(cons)[para]
    step = (span[1]-span[0])/300

    bTable = np.empty((0,6))
    for i in range(len(fixedPoints)):
        fp = np.array(fixedPoints[i],dtype="float64")
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

def plot_aut_bifur():
    fix_list = make_aut_bifur(para=uEmax,span=[0.0,0.04],cons=[(pmax,0.5), (uEmax,0.035), (mE,0.03), (NI,1e-4)])

    unstable = fix_list[:,-1] == False
    stable = fix_list[:,-1] == True
    for i in range(2):
        m, ms, alpha = "o", 3.0, 0.7
        plt.plot(fix_list[unstable,0], fix_list[unstable,2], color = "C2", marker = m, mfc = "none", ms= ms, alpha=alpha, ls="")
        plt.plot(fix_list[stable,0],   fix_list[stable,2],   color = "C2", marker = m,               ms= ms, alpha=alpha, ls="")

        plt.plot(fix_list[unstable,0], fix_list[unstable,1], color = "C0", marker = m, mfc = "none", ms= ms, alpha=alpha, ls="")
        plt.plot(fix_list[stable,0],   fix_list[stable,1],   color = "C0", marker = m,               ms= ms, alpha=alpha, ls="")
    plt.show()

def make_heat_graph(para1, para2, span1, span2, grid_size = 10):
    para1_list = np.linspace(span1[0],span1[1], grid_size)
    para2_list = np.linspace(span2[0],span2[1], grid_size)
    heat_matrix = np.empty(shape=(len(para1_list),len(para2_list)))
    for i in range(len(para1_list)):
        p1 = para1_list[i]
        for j in range(len(para2_list)):
            p2 = para2_list[j]
            cD = makeCons([(para1,p1),(para2,p2)])
            fp_list = _curate_fps(checkFixedPoint(0, 1, cD), [0,1] ,cD)
            print(f" {para1.name} = {p1}, {para2.name} = {p2},  fps = {fp_list}")
            if len(fp_list) == 3:                           # <--- Normal case with three fixed points present
                symb_load = fp_list[1][1]/fp_list[1][0]
            elif len(fp_list) == 2 and fp_list[1][1] > 0:   # <--- Special case where no trivial fixed point exists
                symb_load = fp_list[1][1]/fp_list[1][0]
            elif len(fp_list) == 1:                         # <--- When only trivial fixed point exists we check if it is stable or unstable
                if checkSymbolicStab(fp_list[0],cD):        #      if unstable, the parasitic state should exists a be an attractor. We choose to interpret this as no stable fixed point exists
                    symb_load = 0
                else:
                    symb_load = None
            else:
                symb_load = None   #raise ValueError("Okej, okej now you need to investigate the bifur diagram or somepin!")
            
            heat_matrix[i,j] = symb_load
    
    df = pd.DataFrame(heat_matrix[::-1], index = np.round(para1_list[::-1], 2), columns = np.round(para2_list, 2) )
    print(df)
    ax = sns.heatmap(df, linewidth=0.5, vmin=0, vmax=0.25, cbar_kws={"label": "$E/H$"})

    ax.set_xticks(np.linspace(0.5, heat_matrix.shape[1] - 0.5, 5))           # Attempt at faking a continous axis
    ax.set_xticklabels(np.round( np.linspace(span2[0], span2[1], 5), 3))     #
    ax.set_yticks(np.linspace(0.5, heat_matrix.shape[0] - 0.5, 5))           # 
    ax.set_yticklabels( np.round( np.linspace(span1[1], span1[0], 5), 3 ))   # 
    
    ax.set_ylabel(f"${para1.name}$")
    ax.set_xlabel(f"${para2.name}$")

    #plt.show()
    name1, name2 = para1.name.replace("\\","").replace("{","").replace("}",""), para2.name.replace("\\","").replace("{","").replace("}","")
    plt.savefig(f"figs/heat_graph_{name1}_{name2}.png")


def make_init(ranges = None):
    ranges = ranges or [[0,166], [0,1], [0,0.06], [1e-5,0.22]]
    rand_vec = np.random.rand(4)
    
    y0 = []
    for i in range(len(ranges)):
        val = ranges[i][0] + (ranges[i][1]-ranges[i][0])*rand_vec[i]
        if i == 1:
            val = val*y0[0]
        y0.append(val)
    
    return y0


def plot_area_of_attraction(num_lines = 10, cons = []):
    fps = find_all_fps(cons)
    healthy = fps[-1]
    print(fps)
    mc = "k"
    for i in range(num_lines):
        y0 = make_init([[healthy[0]]*2, [healthy[1]/healthy[0]]*2, [0.0001,0.07], [0.0001,0.22]])
        t, y, funcs, cD = simSystem(y0,[0,1500],cons=cons)
        print(f"{y0}  --->   {y[:,-1]}")
        for fp_num, fp in enumerate(fps):
            diff = np.linalg.norm( y[:,-1] - np.array(fp, dtype="float64"))
            if diff < 1e-1:
                mc = "g" if fp_num == 3 else "r"
                print(fp_num)
                break
        plt.plot(y0[2], y0[3], marker = "o", color=mc)

    for fp in fps:
        plt.plot(fp[2], fp[3], marker = "x", color="k")
    
    plt.ylabel("C")
    plt.xlabel("N")
    plt.show()


if __name__ == "__main__":
    #make_heat_graph(to,pmax,[0.5,2],[0.25,1], 10)

    plot_area_of_attraction(num_lines=200, cons=[(s,1.3)])
    