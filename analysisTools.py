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

from tqdm import tqdm

##### Simulation and basic plotting ############################
p = sns.color_palette("colorblind")
info = pd.DataFrame( 
    data = [["$H$",        "H C-mol/m$^2$",       "Organism biomass", p[0], (1,0)],
            ["$E$",        "E C-mol/m$^2$",       "Organism biomass", p[2], (1,0)],
            ["$N$",        "DIN N-mol/H C-mol",   "Inorganic pool",   p[4], (1,0)],
            ["$C$",        "DIC C-mol/H C-mol",   "Inorganic pool",   p[3], (1,0)],
            ["$E/H$",      "E C-mol/H C-mol",     "Symbiont load",   p[0], (1,0)],

            [r"$\mu_H$",              r"$d^{-1}$",  "Growth rate",             p[0], (1,0)],
            [r"$\mu_E$",              r"$d^{-1}$", "Growth rate",             p[2], (1,0)],
            [r"$\rho_{food}$",        r"$d^{-1}$", "Organic carbon flux",     p[7], (1,0)],
            [r"$\rho_{photo}$",       r"$d^{-1}$", "Organic carbon flux",     p[1], (1,0)],
            ["$p_E$",                 r"$d^{-1}$", "Inorganic carbon flux",   p[2], (2,2)],
            ["$r_H$",                 r"$d^{-1}$", "Inorganic carbon flux",   p[0], (2,2)],
            ["$r_E$",                 r"$d^{-1}$", "Inorganic carbon flux",   p[2], (2,2)],
            ["net $N$ uptake by $H$", r"$d^{-1}$", "Inorganic nutrient flux", p[0], (1,0)],
            ["net $N$ uptake by $H$", r"$d^{-1}$", "Inorganic nutrient flux", p[0], (1,0)],
            ["$e_H$",                 "-",         "Net growth efficiency",   p[0], (1,0)],
            ["$e_E$",                 "-",         "Net growth efficiency",   p[2], (1,0)]],
    columns = ["tex_name", "unit", "type", "color", "dashes"],
    index = ["H", "E", "N", "C", "E/H", "muH", "muE", "rhoDOC", "rhoPhoto", "$pE", "rH", "rE", "netNH", "netNE", "eH", "eE"]
)


def simSystem(y0,tSpan,cons=[],tEval=None):
    cD  = makeCons(cons)
    sol = integ.solve_ivp(endo, y0=y0, t_span=tSpan, t_eval=tEval, args=(cD,), dense_output=False, method="Radau", vectorized=True,
                          max_step = np.inf, rtol=1e-8, atol = 1e-8, events=[symbDeath])
    if sol.status == 1:
        return sol.t_events[0][0], sol.y_events[0][0], makeFuncs(sol.t_events[0][0],sol.y_events[0][0],cD), cD
    if sol.status == -1:
        return None,None,None,None
    funcs = makeFuncs(sol.t,sol.y,cD)
    return sol.t, sol.y, np.array(funcs), cD


def make_df(t, y, funcs, save = ""):
    """Makes a pandas dataframe of a given simulation solution. Save the simulation as csv file if specified
    
    Args:
        t (array-like): Time vector of solution. Will set index for dataframe
        y (array-like): Solution to numerical integration
        funcs (array-like): The collection of function used to solve the ODE (what is returned by make_funcs())

    Returns:
        pandas.DataFrame: Solution given as dataframe
    """
    muH, muE, rhoDOC, rhoPhoto, pE, rH, rE, netNH, netNE, eH, eE = funcs
    columns = ["H", "E", "N", "C", "E/H", r"$\mu_H$", r"$\mu_E$", r"$\rho_{food}$", r"$\rho_{photo}$", "$p_E$",  "net $N$ uptake by $H$", "net $N$ uptake by $E$", "$r_H$", "$r_E$"]
    f = [muH, muE, rhoDOC, rhoPhoto, pE, netNH, netNE, rE, rH]


    data = np.r_[y, [y[1,:]/y[0,:]], f ]
    df = pd.DataFrame(np.transpose(data), index=t, columns=columns)

    if save:
        df.to_csv("sims/sim_" + save)

    return df


def make_df2(t, y, funcs, save = ""):
    """Makes a pandas dataframe of a given simulation solution. Save the simulation as csv file if specified
    
    Args:
        t (array-like): Time vector of solution. Will set index for dataframe
        y (array-like): Solution to numerical integration
        funcs (array-like): The collection of function used to solve the ODE (what is returned by make_funcs())

    Returns:
        pandas.DataFrame: Solution given as dataframe
    """
    
    sim_values = np.r_[y, [y[1]/y[0]], funcs]
    names = info.index.to_numpy()
    data = []
    for name, sol in zip(names, sim_values):
        for time, value in zip(t, sol):
            data.append( {"time": time, "value": value, "name": name, "type": info.loc[name, "type"]} )
    df = pd.DataFrame(data)

    if save:
        df.to_csv("sims/sim_" + save)

    return df


def plot_sim(df, var_list, ax=None, yscale="log", ybottom=None, ytop=None):
    """Plotting the variables of a given simulation in matplotlib"""
    p = sns.color_palette("colorblind")
    color_map = {
        "H": p[0], "E": p[2], "N": p[4], "C": p[3], "E/H": p[1],
        r"$\mu_H$": p[0], r"$\mu_E$": p[2], r"$\rho_{food}$": p[7], r"$\rho_{photo}$": p[1],
        "net $N$ uptake by $H$": p[0], "net $N$ uptake by $E$": p[2], 
        "$p_E$": p[2], "$r_H$": p[0], "$r_E$": p[2]
    }

    data = df[var_list]

    if not ax:
        sim_fig, ax = plt.subplots()

    sns.set_theme(context="talk", style="ticks")
    sns.lineplot(data, palette=color_map, ax=ax, dashes=False)   # dashes = [(1,0), (1,1), osv.]

    ax.set_yscale(yscale)
    ax.set_ylim(bottom=ybottom, top=ytop)
    ctx = sns.plotting_context("talk")
    ax.tick_params(axis="both",labelsize=ctx["axes.labelsize"])
    ax.margins(y=0.05)

    return ax


def plot_sim_2(df, var_list, scale="log", ax=None):

    df = df[df["name"].isin(var_list)]
    palette = [info.loc[var, "color"] for var in var_list] 

    sns.set_theme(context="talk", style="ticks")
    ax = sns.lineplot(df, x="time", y="value", hue="name", palette=palette, ax=ax, legend="brief")
    ax.set_yscale(scale)
    plt.show()



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
    J = J.subs([(H,yStar[0]), (E,yStar[1]), (N,yStar[2]), (C,yStar[3])])

    eig, _ = la.eig(np.array(J.tolist(), dtype=float))
    stab, numDir = True, 0

    for val in eig:
        if sp.re(sp.N(val)) > 0:
            stab = False
            numDir += 1

    return stab #, numDir


def checkIllegal(state,yStar,cD):
    muH, muE, rhoDOC, rhoPhoto, pE, rH, rE, netNH, netNE, eH, eE = makeFuncs(0,yStar,cD)
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


def find_all_fps(cons=[], ignore_H_lim = True):
    """Symbilicly finds all feasible fixed point of the function
    
    Args:
        cons (list): list pair (para, value), where parameter values should be changed from default value to new value
        ignore_H_lim (bool): If the symbolic solver should ignore the possibility of H being N-limited. No feasible fixed points should exist and ignoring this significantly increases the speed

    Returns:
        list: list of fixed points (given as tuples)
    """
    state_list = [(0,0), (0,1)] if ignore_H_lim else [(0,0), (0,1), (1,1), (1,0)]
    cD = makeCons(cons)

    if not ignore_H_lim:
        fps = []
        for i, j in [(0,0), (0,1), (1,1), (1,0)]:
            fps = fps + _curate_fps(checkFixedPoint(i,j,cD), [i,j], cD)
        return fps #OBS unorded here!
    
    fps = [None]*4
    E_not_N_lim = _curate_fps(checkFixedPoint(0,0,cD), [0,0], cD)
    E_N_lim     = _curate_fps(checkFixedPoint(0,1,cD), [0,1], cD)

    i = 0
    for fp in E_not_N_lim:
        if fp[1] == 0.0:
            fps[0] = fp
        else:
            fps[1+i] = fp
            i += 1

    i = 0
    for fp in E_N_lim:
        if fp[1] == 0.0:
            fps[0] = fp
        else:
            fps[2+i] = fp
            i += 1

    return fps


def makeSymbBifur(para,span,cons=[],ignore_H_lim=True): 
    pValues = np.linspace(span[0],span[1],30)
    fixList = []
    try:
        for p in tqdm(pValues,desc=f"Progress for parameter: {para}, in range: {span}", unit="run"):
            new_cons = cons + [(para,p)]
            cD = makeCons(new_cons)
            curated_fps = find_all_fps(new_cons, ignore_H_lim)
            for i, fp in enumerate(curated_fps):
                if fp != None:
                    stab = checkSymbolicStab(fp, cD)
                    fixList.append( np.concatenate([ [p], fp, [i], [stab]]) )
                    tqdm.write(f"{para}: {p}, H: {round(fp[0],20)}, E: {round(fp[1],20)}, N: {round(fp[2],20)}, C: {round(fp[3],20)}, fp: {i}, stable = {stab}")
                    #sys.stdout.flush()
    except KeyboardInterrupt:
        tqdm.write("\n>>>> Interupted by user")
    return np.array(fixList)


##### Bifurcation diagrams by using automatic differentiation #############
def make_jac(f):
    return jax.jacfwd(f)


def check_stab(yStar,cD):
    """Checking stability of fixed point using automatic differentiation
    
    Args:
        yStar (array-like): The fixed point
        cD (dict or dataframe): Dictionary of parametervalues
    
    Returns:
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
    tEnd = 1000                                                                 ## THIS WHOLE SECTION IS OUTDATED! REMOVE IF NOT RECYCLED!!
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

    Args:
        y (array-like): Vector of which oscillation is check (OBS: should be evenly spaced in timesteps)
        tol (float): Tolerence of solution 

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


def make_heat_graph(para1, para2, span1, span2, grid_size = 10, cons=[]):
    para1_list = np.linspace(span1[0],span1[1], grid_size)
    para2_list = np.linspace(span2[0],span2[1], grid_size)
    heat_matrix = np.empty(shape=(len(para1_list),len(para2_list)))
    for i in range(len(para1_list)):
        p1 = para1_list[i]
        for j in range(len(para2_list)):
            p2 = para2_list[j]
            cD = makeCons(cons + [(para1,p1),(para2,p2)])
            fp_list = _curate_fps(checkFixedPoint(0, 1, cD), [0,1] ,cD)
            print(f" {para1.name} = {p1}, {para2.name} = {p2},  fps = {fp_list}")
            if len(fp_list) == 3:                           # <--- Normal case with three fixed points present
                symb_load = fp_list[1][0]
            elif len(fp_list) == 2 and fp_list[1][1] > 0:   # <--- Special case where no trivial fixed point exists
                symb_load = fp_list[1][0]
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
    ax = sns.heatmap(df, linewidth=0.5, vmin=0, cbar_kws={"label": "$H$"}, annot=True, cmap="viridis")  #cmap="magma", "plasma", "cividis"

    ax.set_xticks(np.linspace(0.5, heat_matrix.shape[1] - 0.5, 5))           # Attempt at faking a continous axis
    ax.set_xticklabels(np.round( np.linspace(span2[0], span2[1], 5), 3))     #
    ax.set_yticks(np.linspace(0.5, heat_matrix.shape[0] - 0.5, 5))           # 
    ax.set_yticklabels( np.round( np.linspace(span1[1], span1[0], 5), 3 ))   # 
    
    ax.set_ylabel(f"${para1.name}$")
    ax.set_xlabel(f"${para2.name}$")

    #plt.show()
    name1, name2 = para1.name.replace("\\","").replace("{","").replace("}",""), para2.name.replace("\\","").replace("{","").replace("}","")
    plt.savefig(f"figs/plotted_sims/heat_graph_{name1}_{name2}.png")
    plt.show()


def make_init(num_samples, ranges = None):
    ranges = ranges or [[0,150], [0,1], [0,0.06], [1e-5,0.22]]
    rand_vec = np.random.rand(4)
    
    sample = []
    for j in range(num_samples):
        y0 = []
        for i in range(len(ranges)):
            val = ranges[i][0] + (ranges[i][1]-ranges[i][0])*rand_vec[i]
            if i == 1:
                val = val*y0[0]
            y0.append(val)
        sample.append(y0)
    
    return y0


def lhc_sampling(num_samples, ranges = None):
    ranges = ranges or [[1,150], [0, 0.8], [0, 0.02], [1e-5,0.22]]
    ranges = np.array(ranges)
    sample = stats.qmc.LatinHypercube(d = len(ranges)).random(n=num_samples)
    sample_scaled = stats.qmc.scale(sample, l_bounds = ranges[:,0], u_bounds=ranges[:,1])
    sample_scaled[:,1] = sample_scaled[:,0] * sample_scaled[:,1]
    return sample_scaled


def check_conv(y_end, fps, cons, second_attempt = False):
    """Checks which fixed point a solution has converged to.
    
    Args:
        y_end (list): End-point of a solution.
        fps (list): List of fixed points in known sorted order.

    Returns:
        int: Integer indicating which fixed point the system has converged to.
    """
    for fp_num, fp in enumerate(fps):
        if fp == None: continue
        diff = np.linalg.norm( y_end - np.array(fp, dtype="float64") )
        if diff < 5e-1:
            return fp_num
    if not second_attempt:
        t, y, funcs, cD = simSystem(y_end,[0,5000],cons=cons)
        return check_conv(y[:,-1],fps,cons,second_attempt=True)
    #print(f"Failure to converge after second attempt\n y_end: {y_end}")
    return 0   #return 0 when no convergence happen, species probably die update the count in some way             #raise ValueError("OBS! Solution did not converge to any fixed point in list")


def _to_edges(c):
    dc = np.diff(c) / 2
    edges = np.concatenate(([c[0] - dc[0]], c[:-1] + dc, [c[-1] + dc[-1]]))
    return edges


def plot_area_of_attraction(plot_vars, l_bounds, u_bounds, grid_size = 10, cons = [], healthy_fp = True): # under construction
    x_ind, y_ind = plot_vars
    fps = find_all_fps(cons)
    healthy, unhealthy = fps[-1], fps[0]
    y0 = list(healthy) if healthy_fp == True else list(unhealthy)
    var1_list, var2_list = np.linspace(l_bounds[0], u_bounds[0], grid_size), np.linspace(l_bounds[1], u_bounds[1], grid_size)

    if x_ind in [-1, 4]:
        plt.plot(y0[1]/y0[0], y0[y_ind], "rx")
    elif y_ind in [-1, 4]:
        plt.plot( y0[x_ind], y0[1]/y0[0], "rx")
    else:
        plt.plot(y0[x_ind], y0[y_ind], "rx")


    mat = np.empty(shape=(grid_size,grid_size))

    with tqdm(total=grid_size**2, desc = "Total progress") as pbar:
        for i, var1 in enumerate(var1_list):
            for j, var2 in enumerate(var2_list):
                pbar.update(1)
                if x_ind != -1 and x_ind !=4:
                    y0[x_ind] = var1
                else:
                    y0[1] = y0[0]*var1
                if y_ind != -1 and y_ind != 4: 
                    y0[y_ind] = var2 
                else: 
                    y0[1] = y0[0]*var2
                
                t, y, funcs, cD = simSystem(y0,[0,1500],cons=cons)
                fp_num = check_conv(y[:,-1],fps,cons)
                mat[i,j] = fp_num // 2


    print(mat)
    plt.pcolormesh(_to_edges(var1_list), _to_edges(var2_list), np.transpose(mat), cmap="Greens", shading="auto", vmin=0)
    plt.colorbar()

   
    label_list = ["H", "E", "N", "C", "E/H"]
    plt.xlabel(label_list[x_ind])
    plt.ylabel(label_list[y_ind])
    plt.show()


def prob_of_states(n, plot_vars, ranges = None, cons = [], save_fig=True, grid_size = 40):
    x_ind, y_ind = plot_vars
    fps = find_all_fps(cons)
    print(fps)
    healthy, unhealthy = fps[-1], fps[1]

    init_list = lhc_sampling(n, ranges)
    num_healthy, mat = 0, []
    color_list = ["C1", "C0"]

    for y0 in tqdm(init_list):
        t, y, funcs, cD = simSystem(y0,[0,1000],cons=cons)
        fp_num = check_conv(y[:,-1],fps,cons)

        x = y0[x_ind] if (x_ind != -1 and x_ind != 4) else y0[1]/y0[0]
        y = y0[y_ind] if (y_ind != -1 and y_ind != 4) else y0[1]/y0[0]

        mat.append([x, y, fp_num//2])

        if fp_num == 3: num_healthy += 1
        plt.plot(x, y, marker = "o", ls = "", color = color_list[fp_num//2], alpha = 0.2)
        
    
    print(f"Probability of convergence to healthy state {num_healthy/n}")
    label_list = ["H", "E", "N", "C", "E/H"]
    plt.xlabel(label_list[x_ind])
    plt.ylabel(label_list[y_ind])

    for fp, name in [(healthy,"Healthy"), (unhealthy,"Unhealthy")]:
        x_star = fp[x_ind] if (x_ind != -1 and x_ind != 4) else fp[1]/fp[0]
        y_star = fp[y_ind] if (y_ind != -1 and y_ind != 4) else fp[1]/fp[0]
        plt.plot(x_star, y_star, marker = "x", color = "k")
        plt.text(x_star*1.01, y_star*1.01, name+" state")

        if x_ind in [-1, 4] or y_ind in [-1,4]:
            break


    data = np.array(mat)                                          ### OBS: edit to more readable names 
    x, y, z = data[:,1], data[:,0], data[:,2]
    sum_v, xedges, yedges = np.histogram2d(x, y, bins=grid_size, weights=z)
    count, _, _ = np.histogram2d(x, y, bins=[xedges, yedges])
    avg = np.divide(sum_v, count, out=np.zeros_like(sum_v), where=count>0)

    plt.figure()
    plt.pcolormesh(yedges, xedges, avg, cmap="Greens",vmin=0,vmax=1)  #.imshow(avg, interpolation='bilinear') 
    plt.colorbar(label="Probalility")

    for fp, name in [(healthy,"Healthy"), (unhealthy,"Unhealthy")]:
        x_star = fp[x_ind] if (x_ind != -1 and x_ind != 4) else fp[1]/fp[0]
        y_star = fp[y_ind] if (y_ind != -1 and y_ind != 4) else fp[1]/fp[0]
        plt.plot(x_star, y_star, marker = "x", color = "k")
        plt.text(x_star*1.01, y_star*1.01, name+" state")

        if x_ind in [-1, 4] or y_ind in [-1,4]:
            break


    plt.xlabel(label_list[x_ind])
    plt.ylabel(label_list[y_ind])

    if save_fig:
        name = ""
        for p, v in cons:
            name = name + f"_{p.name}={v}"
        plt.savefig("figs/plotted_sims/aoa" + name + ".png" ,dpi=300, bbox_inches="tight")
        np.savetxt("sims/aoa_" + name + "_raw.txt", data)

    plt.show()
    return


if __name__ == "__main__":
    #make_heat_graph(s,uEmax,[1,1.5],[0.0001,0.06], 10, cons = [(pmax, 0.45), (uEmax,0.033), (uHmax,0.0045),
#
    #     (KCO2, 0.02), (KNE, 0.03),   (KNH, 0.0001),
#
    #     (NI, 0.001), (CI, 0.09)])  

    #plot_area_of_attraction([2,-1], [1e-5, 0.01], [0.06, 0.5], grid_size=10, cons=[(s,1.35)]) #,(mE,0.06),(KNE,0.01),(CI,0.13)])

    #prob_of_states(30000, [2, -1], cons=[(s,1.0)], save_fig=True)

    t,y,funcs,cD = simSystem([25,0.01, 0.001, 0.01], [0,500])
    df = make_df2(t,y,funcs, save="single_sim")
    
    plot_sim_2(df, ["H", "E"])

    

