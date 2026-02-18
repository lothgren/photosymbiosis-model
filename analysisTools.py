#Tools to analyse coral model (created 20/6)

from model import *

import matplotlib.pyplot as plt
import matplotlib.gridspec as gspec
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
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

##### Simulation and saving data ############################
p = sns.color_palette("colorblind")
info = pd.DataFrame( 
    data = [["$H$",                  r"mol C$_\text{O}$/m$^2$",     "Organism biomass",         p[0], (1,0)],
            ["$E$",                  r"mol C$_\text{O}$/m$^2$",     "Organism biomass",         p[2], (1,0)],
            ["$N$",                  r"mol DIN/mol C$_\text{O}$", "Inorganic pool",           p[4], (1,0)],
            ["$C$",                  r"mol DIC/mol C$_\text{O}$", "Inorganic pool",           p[3], (1,0)],
            ["$E/H$",                "molar ratio",   "Symbiont load",            p[1], (1,0)],

            [r"$\mu_H$",              r"$d^{-1}$",        "Host carbon flux",         p[0], (1,0)],
            [r"$\mu_E$",              r"$d^{-1}$",        "Endosymbiont carbon flux", p[2], (1,0)],
            [r"$\rho_{food}$",        r"$d^{-1}$",        "Host carbon flux",         p[7], (1,0)],
            [r"$\rho_{photo}$",       r"$d^{-1}$",        "Host carbon flux",         p[1], (1,0)],
            ["$p_E$",                 r"$d^{-1}$",        "Endosymbiont carbon flux", p[2], (2,2)],
            ["$r_H$",                 r"$d^{-1}$",        "Inorganic carbon flux",    p[0], (2,2)],
            ["$r_E$",                 r"$d^{-1}$",        "Inorganic carbon flux",    p[2], (2,2)],
            ["net $N$ uptake by $H$", r"$d^{-1}$",        "Inorganic nutrient flux",  p[0], (1,0)],
            ["net $N$ uptake by $E$", r"$d^{-1}$",        "Inorganic nutrient flux",  p[2], (1,0)],
            ["$e_H$",                 "-",                "Net growth efficiency",    p[0], (1,0)],
            ["$e_E$",                 "-",                "Net growth efficiency",    p[2], (1,0)]],
    columns = ["tex_name", "unit", "type", "color", "dashes"],
    index = ["H", "E", "N", "C", "E/H", "muH", "muE", "rhoDOC", "rhoPhoto", "pE", "rH", "rE", "netNH", "netNE", "eH", "eE"]
)


def sim_system(y0,t_span,cons=[],t_eval=None):
    """
    Numerical simulation with picked arguments fitting for the ODE fiunction "endo"
    
    Args:
        y0 (array-like): Initial values of the numerical simulation
        t_span (array-like): Time span of numerical simulation 
        cons (dict or DataFrame, optional): Parameters values that should be changed in the base set for this specific simulation
        t_eval (array-like, optional): Specific evalutaion points for the simulation

    Returns:
        tuple: Tuple returning the content of the simulation and total parameter set. Return (None)*4 if simulation failed
    """
    cD  = make_cons(cons)
    sol = integ.solve_ivp(endo, y0=y0, t_span=t_span, t_eval=t_eval, args=(cD,), dense_output=False, method="Radau", vectorized=True,
                          max_step = np.inf, rtol=1e-8, atol = 1e-8, events=[symb_death, load_stop])
    if sol.status == 1:
        return sol.t_events[0][0], sol.y_events[0][0], make_funcs(sol.t_events[0][0],sol.y_events[0][0],cD), cD
    if sol.status == -1:
        return (None)*4
    funcs = make_funcs(sol.t,sol.y,cD)
    return sol.t, sol.y, np.array(funcs), cD


##### Symbolic function ####################################################
def check_for_fps(H_limited, E_limited, cD):
    """
    Function uses symbolic computation to find fixed points for a given state (H and/or E under nutrient limitation)

    Args:
        H_limited (bool): True if H is nutrient limited
        E_limited (bool): True if E is nutrient limited
        cD (dict or DataFrame): Complete list of parameters for whitch the fixed points should be found

    Returns:
        list[tuple]: List of fixed points
    """

    [dH,dE,dN,dC] = endo_symbolic(H_limited,E_limited)
    f_subbed = [dH.subs(cD),dE.subs(cD),dN.subs(cD),dC.subs(cD)]
    fps = sp.solve(f_subbed,[H, E, N, C])
    #print(fixedPoints)
    return fps


def check_symb_stab(yStar, cD):
    """Check stability of fixed point by symbolically calculation jacobian and eigenvalues
    
    Arguments:
    yStar: array-like, giving the fixed point of the system
    cD: dict or dataframe, giving the parameter values of the system

    Returns
    True if all eigenvalues have negative real part, False otherwise
    """
    funcs = make_funcs(np.nan,yStar,cD)
    state= [funcs[-2]<1/(1+cD[s]),  funcs[-1]<1/(1+cD[s])]

    F = sp.Matrix(endo_symbolic(state[0],state[1])).subs(cD)
    J = F.jacobian([H,E,N,C])
    J = J.subs([(H,yStar[0]), (E,yStar[1]), (N,yStar[2]), (C,yStar[3])])

    eig, _ = la.eig(np.array(J.tolist(), dtype=float))
    stab, numDir = True, 0

    for val in eig:
        if sp.re(sp.N(val)) > 0:
            stab = False
            numDir += 1

    return stab #, numDir


def check_illegal(state,yStar,cD):
    muH, muE, rhoDOC, rhoPhoto, pE, rH, rE, netNH, netNE, eH, eE = make_funcs(0,yStar,cD)
    currState = [bool(eH<(1/(cD[s]+1))) , bool(eE<(1/(cD[s]+1)))]
    return currState == state


def _key(n):
    return sum(n)


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
        if all(val>=0 for val in fp) and check_illegal(state,fp,cD): 
            curated_fps.append(fp)
    return sorted(curated_fps,key=_key,reverse=True)


def find_all_fps(cons=[], ignore_H_lim = True):
    """Symbolicly finds all feasible fixed point of the function
    
    Args:
        cons (list): list pair (para, value), where parameter values should be changed from default value to new value
        ignore_H_lim (bool): If the symbolic solver should ignore the possibility of H being N-limited. No feasible fixed points should exist and ignoring this significantly increases the speed

    Returns:
        list: list of fixed points (given as tuples)
    """
    cD = make_cons(cons)

    if not ignore_H_lim:                           ## This should maybe be gone through once more
        fps = []
        for i, j in [(0,0), (0,1), (1,1), (1,0)]:
            fps = fps + _curate_fps(check_for_fps(i,j,cD), [i,j], cD)
        return fps #OBS unorded here!
    
    fps = [None]*4
    E_not_N_lim = _curate_fps(check_for_fps(0,0,cD), [0,0], cD)
    E_N_lim     = _curate_fps(check_for_fps(0,1,cD), [0,1], cD)

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
    pValues = np.linspace(span[0],span[1],30)   ## <---------------
    fixList = []
    fp_names = ["Trivial", "Parasitic", "Healthy", "Healthy"]
    try:
        for p in tqdm(pValues,desc=f"Progress for parameter: {para}, in range: {span}", unit="run"):
            new_cons = cons + [(para,p)]
            cD = make_cons(new_cons)
            curated_fps = find_all_fps(new_cons, ignore_H_lim)
            for i, fp in enumerate(curated_fps):
                if fp != None:
                    stab = check_symb_stab(fp, cD)
                    fixList.append([para] + [p] + list(fp) + [i] + [stab] )
                    tqdm.write(f"{para}: {p}, H: {round(fp[0],20)}, E: {round(fp[1],20)}, N: {round(fp[2],20)}, C: {round(fp[3],20)}, fp: {i}, stable = {stab}")
                    #sys.stdout.flush()
    except KeyboardInterrupt:
        tqdm.write("\n>>>> Interupted by user")
    return fixList


##### Bifurcation diagrams by using automatic differentiation #############  Needed to go through once more before publication 
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
        cD = make_cons(cons+[(para,p)])
        f = lambda y: endo(0,y,cD,minFunc=min_approx)
        J = make_jac(f)
        sol = opt.root(f, y0, method="hybr", tol=1e-10, options={"xtol":1e-10,"maxfev":0,"eps":0.1})
        if sol.success and all(val>=0 for val in sol.x):
            fixList.append(np.concatenate( [[p], sol.x, [fp_num] ,[check_stab(sol.x,cD)]]))
        else:
            fixList.append( [p] + [np.nan]*len(sol.x) + [fp_num] + [False])
        y0 = sol.x
    return fixList


def make_aut_bifur(para,span,cons=[]):
    cD = make_cons(cons)
    standardVal = cD[para]
    step = (span[1]-span[0])/200

    curatedFps = []
    for state in [[0,0], [0,1] ]:     #, [1,1], [1,0]]:
        fixedPoints = sorted(check_for_fps(state[0],state[1],cD),key=_key,reverse=True)
        for fp in fixedPoints:
            if all(val>=0 for val in fp) and check_illegal(state,fp,cD): 
                curatedFps.append(np.array(fp,dtype="float64"))
    curatedFps = sorted(curatedFps,key=_key)

    fix_list = []
    for i in range(len(curatedFps)):
        fp = curatedFps[i]
        for j in range(2):
            pList = np.arange(standardVal,span[j],(-1)**(j+1)*step)
            fix_list = fix_list + aut_bifur(para,pList,fp,fp_num=i,cons=cons)
    
    return np.array(fix_list)


##### Bifurcation diagrams by numerically solving for the fixed point #####  IF above is correctly implemented lets remove this
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
        cD = make_cons(cons+[(para,p)])
        sol = opt.root(lambda y: endo(0,y,cD,minFunc=min_approx), y0, method="hybr", tol=1e-12, options={"xtol":1e-12,"maxfev":0,"eps":0.1})
        if sol.success and all(val>=0 for val in sol.x):
            stab = checkStab(sol.x, cD)
            fixList.append( np.concatenate([ [p],y0,[stab] ]) )
            print(f"{para}: {p}, H: {round(sol.x[0],3)}, E: {round(sol.x[1],10)}, N: {round(sol.x[2],10)}, C: {round(sol.x[3],10)}, stable = {stab}")
        y0 = sol.x
    return np.array(fixList)


def makeNumBifur(para,span,cons=[]):
    cD = make_cons(cons)
    fixedPoints = check_for_fps(0,1,cD) + check_for_fps(0,0,cD) # check_for_fps(1,0,cD) + check_for_fps(1,1,cD)
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
    standardVal = make_cons(cons)[para]
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



###### Saving simulation data ################################################
def make_df(t, y, funcs, save_name = ""):
    """Makes a pandas dataframe of a given simulation solution. Save the simulation as csv file if specified
    
    Args:
        t (array-like): Time vector of solution. Will set index for dataframe
        y (array-like): Solution to numerical integration
        funcs (array-like): The collection of function used to solve the ODE (what is returned by make_funcs())
        save_name (str, optional): If non-empty saves dataframe as csv in sims folder as save_name.csv

    Returns:
        DataFrame: Solution given as dataframe
    """
    columns = info["tex_name"].to_list()

    data = np.r_[y, [y[1,:]/y[0,:]], funcs ]
    df = pd.DataFrame(np.transpose(data), index=t, columns=columns)

    if save_name:
        df.to_csv("sims/sim_" + save_name)

    return df


def save_bifur_data(paras=None, cons=[], save_name=""):
    para_spans = { s: [1,1.4], to: [1,1.25], pmax: [0.01,1.0], uEmax: [0.005,0.09], uHmax: [0.0,0.01], KNE: [0.0001,0.12], KNH: [0.0,0.002], 
            delC: [0.0,0.7], CI: [0.0,0.15], delN: [0.0,0.5], NI: [0.0,0.002], mH: [0.01,0.05], mE: [0.03,0.2], KCO2: [0.0001,0.06], rho0: [0,0.1],
            QFood: [0,0.2], QE: [0.01,0.2], QH: [0.01,0.2], b: [0.01, 1], eps: [0,0.1] }
    subset = dict( [(p, para_spans[p]) for p in paras]) if paras else para_spans

    data = []
    for p, span in subset.items():
        data = data + makeSymbBifur(p, span, cons, ignore_H_lim=True)
    
    df = pd.DataFrame(data, columns=["para_name", "para_value", "H", "E", "N", "C", "fp_num", "stable"])
    df_long = df.melt(id_vars=["para_name", "para_value", "fp_num", "stable"], var_name="var_name", value_name="var_value")

    if save_name:
        df_long.to_csv("sims/"+save_name, index=False)
    return df_long


def make_heat_graph(para1, para2, span1, span2, grid_size = 10, cons=[]):
    para1_list = np.linspace(span1[0],span1[1], grid_size)
    para2_list = np.linspace(span2[0],span2[1], grid_size)
    heat_matrix = np.empty(shape=(len(para1_list),len(para2_list)))
    with tqdm(total=grid_size**2, desc = "Total progress") as pbar:
        for i in range(len(para1_list)):
            p1 = para1_list[i]
            for j in range(len(para2_list)):
                pbar.update(1)
                p2 = para2_list[j]
                cD = make_cons(cons + [(para1,p1),(para2,p2)])
                fp_list = _curate_fps(check_for_fps(0, 1, cD), [0,1] ,cD)
                tqdm.write(f" {para1.name} = {p1}, {para2.name} = {p2},  fps = {fp_list}")
                if len(fp_list) == 3:                           # <--- Normal case with three fixed points present
                    symb_load = fp_list[1][0]
                elif len(fp_list) == 2 and fp_list[1][1] > 0:   # <--- Special case where no trivial fixed point exists
                    symb_load = fp_list[1][0]
                elif len(fp_list) == 1:                         # <--- When only trivial fixed point exists we check if it is stable or unstable
                    if check_symb_stab(fp_list[0],cD):        #      if unstable, the parasitic state should exists a be an attractor. We choose to interpret this as no stable fixed point exists
                        symb_load = 0
                    else:
                        symb_load = None
                else:
                    symb_load = None   #raise ValueError("Okej, okej now you need to investigate the bifur diagram or somepin!")
                
                heat_matrix[i,j] = symb_load
        
    df = pd.DataFrame(heat_matrix[::-1], index = np.round(para1_list[::-1], 2), columns = np.round(para2_list, 2) )
    print(df)

    ax = sns.heatmap(df, linewidth=0.5, vmin=0, cbar_kws={"label": "$H$"}, annot=False, cmap="viridis")  #cmap="magma", "plasma", "cividis"

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


def make_2D_bifur(para1, para2, span1, span2, grid_size = 10, base_cons=[], save_name=""):
    """
    Creates and saves a long format-dataframe with all fixed point values and their stability over a range of 
    two parameters. The dataframe is used in the plotting of the 2D bifurcation diagrams. Specify save_name to  
    save to csv.
    
    Args:
        para1 (sp.Symbol): The sympy symbol of the first parameter
        para2 (sp.Symbol): The sympy symbol of the second parameter
        span1 (list): Span of first parameter
        span2 (list): Span of second parameter
        grid_size (int): Number of sample to generate of each parameter span. Default is 10 
        base_cons (list): Base set of parameter changes. Default is empty
        save_name (str): Name to save dataframe under. Default is empty
    
    Returns:
        pd.DataFrame: the data of the 2D bifurcation diafram in long format
    """
    para1_list = np.linspace(span1[0],span1[1], grid_size)
    para2_list = np.linspace(span2[0],span2[1], grid_size)
    data = []
    with tqdm(total=grid_size**2, desc = f"2D bifurcation progress ({para1}, {para2})") as pbar:
        for i in range(len(para1_list)):
            p1 = para1_list[i]
            for j in range(len(para2_list)):
                pbar.update(1)
                p2 = para2_list[j]
                cons = base_cons + [(para1,p1),(para2,p2)]
                fps = find_all_fps(cons)
                
                for fp_num, fp in enumerate(fps):
                    if fp == None: 
                        fp = [np.nan]*4
                        stab = False
                    else:
                        stab = check_symb_stab(fp, make_cons(cons))
                                                                                
                    for value, name in zip(fp, ["H", "E", "N", "C"]):
                        data.append( [p1, p2, value, fp_num, name, stab] )

    df = pd.DataFrame(data, columns=[f"${para1.name}$", f"${para2.name}$", "value", "fp_num", "name", "stability"])
    if save_name:
        df.to_csv("sims/2D_bifur_"+save_name+".csv", index=False)

    return df


def _lhc_sampling(num_samples, ranges = None):
    ranges = ranges or [[1,120], [0, 1], [0, 0.03], [1e-5,0.22]]
    ranges = np.array(ranges)
    sample = stats.qmc.LatinHypercube(d = len(ranges)).random(n=num_samples)
    sample_scaled = stats.qmc.scale(sample, l_bounds = ranges[:,0], u_bounds=ranges[:,1])
    sample_scaled[:,1] = sample_scaled[:,0] * sample_scaled[:,1]
    return sample_scaled


def _check_conv(y_end, fps, cons, second_attempt = False):
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
        t, y, funcs, cD = sim_system(y_end,[0,5000],cons=cons)
        return _check_conv(y[:,-1],fps,cons,second_attempt=True)
    #print(f"Failure to converge after second attempt\n y_end: {y_end}")
    return 0   #return 0 when no convergence happen, species probably die update the count in some way             #raise ValueError("OBS! Solution did not converge to any fixed point in list")


def prob_of_states(n, ranges = None, cons = [], save_name=False):
    fps = find_all_fps(cons)
    init_list = _lhc_sampling(n, ranges)
    mat = []

    for y0 in tqdm(init_list):
        t, y, funcs, cD = sim_system(y0,[0,1000],cons=cons)
        fp_num = _check_conv(y[:,-1],fps,cons)
        mat.append( np.append(y0, [fp_num//2]))

    if save_name:
        name = save_name
    else:
        name = ""
        for p, v in cons:
            name = name + f"_{_saveable_name(p)}={v}"
    np.savetxt("sims/aoa_" + name + "_raw.txt", mat)
    
    return mat


###### Plotting tools ########################################################
def _saveable_name(para):
    return para.name.replace("\\","").replace("{","").replace("}","")


def plot_aut_bifur(): #still needs work
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


def plot_sim(df, var_list, ax=None, yscale="log", ybottom=None, ytop=None):
    """Plots specified variable in line plot
    
    Args:
        df (DataFrame): Dataframe solution of simulation.
        var_list (list[str]): List of variables/functions that should be plotted written according to the index short names.
        ax (matplotlib.axes.Axes, optional): Specific subplot ax. If not specified then creates nex figure and subplot.
        yscale (str, optional): Y-axis scale. Can be "log" or "linear".
        ybottom (float, optional): Lower y-axis limit.
        ytop (float, optional): Upper y-axis limit.

    Returns:
        matplotlib.axes.Axes: Ax one which the system was plotted.
    """
    palette = [info.loc[var, "color"] for var in var_list]
    var_list = [info.loc[var, "tex_name"] for var in var_list]
    data = df[var_list]

    if not ax:
        sim_fig, ax = plt.subplots()

    sns.set_theme(context="paper", style="ticks")
    sns.lineplot(data, palette=palette, ax=ax, dashes=False, legend="brief")   # dashes = [(1,0), (1,1), osv.]

    ax.set_yscale(yscale)
    ax.set_ylim(bottom=ybottom, top=ytop)
    ctx = sns.plotting_context("paper")
    ax.tick_params(axis="both",labelsize=ctx["axes.labelsize"])
    ax.margins(y=0.05)

    return ax


def plot_bifur(path, paras, vars = ["H", "E"], save_name = ""):
    """
    Plots bifurcation diagram for one or more variables for specified parameters. If more parameters are given puts plots together in a comparable figure.
    
    Args:
        path (str): Path to csv dataframe (long format) with bifurcation data.
        paras (list[str]): List of which parameters should be plotted.
        vars (list[str], optional): Which variables fixed points should be displayed in the bifuraction plot. Defualt is H and E
        save_name (str, optional): If non-empty saves figure as png in figs/bifur under save_name.

    Returns:
        sns.FacetGrid: Grid of bifurcation subplots.
    """
    if isinstance(path, str):
        df = pd.read_csv(path)
    else:
        df = path

    ## Curating df
    subset = df[df["para_name"].isin([str(p) for p in paras] )]
    print(str(paras[0]))
    subset = subset[subset["var_name"].isin(vars)]

    ## Setting up the plot
    num_col = round(len(paras)**(1/2))
    sns.set_theme(style="ticks",context="paper")
    palette = [info.loc[var, "color"] for var in vars]
    markers = {0: "^", 1: "D", 2: "o", 3: "s"}
    g = sns.relplot(subset, x="para_value", y="var_value", hue="var_name", col="para_name", col_wrap=num_col, col_order=[str(p) for p in paras],
                    style="fp_num", palette=palette, markers=markers, dashes=False, kind="line", facet_kws=dict(sharex=False))
    

    ## Adding labels
    legend = g._legend
    legend.texts[0].set_text("Variables")
    legend.texts[0].set_fontweight("bold")
    legend.texts[len(vars)+1].set_text("Fixed point")
    legend.texts[len(vars)+1].set_fontweight("bold")

    g.set_titles("")
    g.set_ylabels(info.loc[vars[0], "unit"])
    for ax, xlabel in zip(g.axes.flat, [p.name for p in paras]):
        ax.set_xlabel(f"${xlabel}$")

    ## Saving 
    if save_name:
        plt.savefig("figs/bifurs/" + save_name + ".png")

    return g


def _subplot_bifur(data, color, **kws):
    ax = plt.gca()

    # stable points – filled
    sns.scatterplot(
        data=data[data["stable"] == 1],
        x="para_value",
        y="var_value",
        style="fp_num",
        hue="var_name",
        legend=False,
        ax=ax
    )


def plot_bifur2(path, paras, vars = ["H", "E"], save_name = ""):
    if isinstance(path, str):
        df = pd.read_csv(path)
    else:
        df = path

    ## Curating df
    subset = df[df["para_name"].isin([str(p) for p in paras] )]
    print(str(paras[0]))
    subset = subset[subset["var_name"].isin(vars)]

    ## Setting up the plot
    num_col = round(len(paras)**(1/2))
    sns.set_theme(style="ticks",context="paper")
    palette = [info.loc[var, "color"] for var in vars]
    markers = {0: "^", 1: "D", 2: "o", 3: "s"}

    g = sns.FacetGrid(subset, hue="var_name", col="para_name", col_wrap=num_col, col_order=[str(p) for p in paras],
                palette=palette, sharex=False)
    
    g.map_dataframe(_subplot_bifur)
    g.add_legend(title="var_name / fp_num")



def plot_2D_bifur(path, var, vmax=None, ax=None, save_name="", cbar=False, cmap="plasma"):
    """
    Plots 2D bifurcation diagram of given dataframe. Dataframe should have been created using make_2D_bifur()-function
    
    Args:
        path (str or pd.DataFrame): Path to load dataframe or actual dataframe
        var (str): Variable whos value are plotted. Either "H", "E", "N" or "C"
        vmax (float): Maximum value for color scale. If not specified maximum value of data is used
        ax (plt.axes.Axes): Ax subplot of figure. If not specified creates an new fixure and subplot
        save_name (str): Name that figure should be saved under. If not specified, it is not saved
        cbar (bool): If a color bar should be included in seaborns heatmap plot. Default is Fales
        cmap (str): If a specific color scale should be used. Default is plasma

    Returns:
        plat.axes.Axes: subplot ax that heatmap was plotted in
    """
    if isinstance(path,str):
        df = pd.read_csv(path)
    else:
        df = path

    cols = df.columns
    df_fp3 = df[(df["name"]==var) & (df["fp_num"]==3)].pivot(index=cols[0], columns=cols[1], values="value")  # non-zero N-lim state
    df_fp1 = df[(df["name"]==var) & (df["fp_num"]==1)].pivot(index=cols[0], columns=cols[1], values="value")  # non-zero C-lim state
    df_fp0 = df[(df["name"]==var) & (df["fp_num"]==0) & (df["stability"]==1)].pivot(index=cols[0], columns=cols[1], values="value") # Semi trivial fixedpoint that is stable

    if var == "E":
        annot_df = ( (df_fp3>df_fp1) | (df_fp3.notna() & df_fp1.isna()) ).replace({True:"*", False:""})
        annot_df = annot_df[::-1]
    else:
        annot_df = None
    sub_df = df_fp3.fillna(df_fp0)

    if not ax:
        fig, ax = plt.subplots(1,1,figsize=(5.5, 4),constrained_layout=True)
        cbar = True

    sns.heatmap(sub_df.iloc[::-1], linewidth=0.5, vmin=0, vmax=vmax, annot=annot_df, fmt="", cbar_kws={"label": info.loc[var, "unit"]}, cmap=cmap, ax=ax, cbar=cbar)  #cmap="magma", "plasma", "cividis"
    ax.axhline(y=5, color="red", ls="--")
    ax.set_xticks(ax.get_xticks()[::2])
    ax.set_yticks(ax.get_yticks()[::2])
    ax.set_xticklabels([f"{x:.2f}" for x in sub_df.columns[::2]], rotation=45)
    ax.set_yticklabels([f"{x:.2f}" for x in sub_df.index[::-2]])

    if save_name:
        plt.savefig(f"figs/bifurs/2D_bifur_{save_name}.png", dpi=300, bbox_inches="tight")
    
    return ax


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
                
                t, y, funcs, cD = sim_system(y0,[0,1500],cons=cons)
                fp_num = _check_conv(y[:,-1],fps,cons)
                mat[i,j] = fp_num // 2


    print(mat)
    plt.pcolormesh(_to_edges(var1_list), _to_edges(var2_list), np.transpose(mat), cmap="Greens", shading="auto", vmin=0)
    plt.colorbar()

   
    label_list = ["H", "E", "N", "C", "E/H"]
    plt.xlabel(label_list[x_ind])
    plt.ylabel(label_list[y_ind])


def plot_aoa(path, plot_vars, grid_size = 30, save_name = "", ax=None):
    if isinstance(path, str):
        data = np.loadtxt(path, dtype="float64")
    else:
        data = np.array(path)

    x_ind, y_ind = plot_vars                                          ### OBS: edit to more readable names 
    x = data[:,y_ind] if not y_ind in [-1, 4] else data[:,1]/data[:,0]
    y = data[:,x_ind] if not x_ind in [-1, 4] else data[:,1]/data[:,0]
    z = data[:,-1]
    sum_v, xedges, yedges = np.histogram2d(x, y, bins=grid_size, weights=z)
    count, _, _ = np.histogram2d(x, y, bins=[xedges, yedges])
    avg = np.divide(sum_v, count, out=np.zeros_like(sum_v), where=count>0)

    if not ax:
        fig, ax = plt.subplots(1,1)
    im = ax.pcolormesh(yedges, xedges, avg, cmap="Greens",vmin=0,vmax=1)  #.imshow(avg, interpolation='bilinear') 
    #fig.colorbar(im, label="Probalility")


    label_list = ["H", "E", "N", "C", "E/H"]

    ax.set_xlabel(label_list[x_ind])
    ax.set_ylabel(label_list[y_ind])

    if save_name:
        plt.savefig("figs/plotted_sims/" + save_name + ".png")
    return im



if __name__ == "__main__":  


    #plot_bifur2("sims/bifur_df", [s,uEmax,pmax])
    #df = make_2D_bifur(s, pmax, [1, 1.5], [0.001, 1], grid_size=15, save_name="test")
    #plot_2D_bifur(df, "E")

    #plot_2D_bifur(path="sims/2D_bifur_s_uEmax.csv", vmax=37.7, var="E", cmap="viridis", save_name="E_uEmax")
    #plot_2D_bifur(path="sims/2D_bifur_s_pmax.csv",  vmax=37.7, var="E", cmap="viridis", save_name="E_pmax")
    #plot_2D_bifur(path="sims/2D_bifur_s_uEmax.csv", vmax = 112.4, var="H"             , save_name="H_uEmax")
    #plot_2D_bifur(path="sims/2D_bifur_s_pmax.csv",  vmax = 112.4, var="H"             , save_name="H_pmax" )
    plot_area_of_attraction([0,-1], [1, 0.01], [110, 1], grid_size=10, cons=[(uEmax,0.0325)])     

    #prob_of_states(30, save_name="test")
    #plot_aoa("sims/aoa_test_raw.txt", [0,-1])
    plt.show()