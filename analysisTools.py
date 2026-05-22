#################################################################
### This script contains functions used to evaluted the model ###
#################################################################

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
    data = [["$H$",                   "$H$",                       r"mol C$_\text{org}$/m$^2$",     "Organism biomass",         p[0], (1,0)],
            ["$S$",                   "$S$",                       r"mol C$_\text{org}$/m$^2$",     "Organism biomass",         p[2], (1,0)],
            ["$N$",                   "$N$",                       r"mol DIN/mol C$_\text{org}$",   "Inorganic pool",           p[4], (1,0)],
            ["$C$",                   "$C$",                       r"mol DIC/mol C$_\text{org}$",   "Inorganic pool",           p[3], (1,0)],
            ["$S/H$",                 "$S/H$",                     "-",           "Symbiont load",            p[1], (1,0)],

            [r"$\mu_H$",              "$H$ GR",                    "1/d",                  "Host carbon flux",         p[0], (1,0)],
            [r"$\mu_S$",              "$R$ GR",                    "1/d",                  "Endosymbiont carbon flux", p[2], (1,0)],
            ["$f$",                   "HFR",                       "1/d",                  "Host carbon flux",         p[7], (1,0)],
            ["$p_H$",                 "PAR",                       "1/d",                  "Host carbon flux",         p[1], (1,0)],
            ["$p_S$",                 "DIC fication rate",         "1/d",                  "Endosymbiont carbon flux", p[2], (2,2)],
            ["$r_H$",                 "$H$ resp. rate",            r"mol DIC/mol C$_\text{org}$/d",                  "Inorganic carbon flux",    p[0], (2,2)],
            ["$r_S$",                 "$S$ resp. rate",            "1/d",                  "Inorganic carbon flux",    p[2], (2,2)],
            ["$H$ net DIN upt.",      "$H$ net DIN upt.",          r"mol DIN/mol C$_\text{org}$/d",                  "Inorganic nutrient flux",  p[0], (1,0)],
            ["$S$ net DIN upt.",      "$S$ net DIN upt.",          "1/d",                  "Inorganic nutrient flux",  p[2], (1,0)],
            ["$e_H$",                 "$H$ net growth efficiancy", "-",                             "Net growth efficiency",    p[0], (1,0)],
            ["$e_S$",                 "$S$ net growth efficiancy", "-",                             "Net growth efficiency",    p[2], (1,0)]],
    columns = ["tex_symbol", "tex_name", "unit", "type", "color", "dashes"],
    index = ["H", "S", "N", "C", "S/H", "muH", "muS", "rhoDOC", "rhoPhoto", "pS", "rH", "rS", "netNH", "netNS", "eH", "eS"]
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
def check_for_fps(H_limited, S_limited, cD):
    """
    Function uses symbolic computation to find fixed points for a given state (H and/or S under nutrient limitation)

    Args:
        H_limited (bool): True if H is nutrient limited
        S_limited (bool): True if S is nutrient limited
        cD (dict or DataFrame): Complete list of parameters for whitch the fixed points should be found

    Returns:
        list[tuple]: List of fixed points
    """

    [dH,dS,dN,dC] = endo_symbolic(H_limited,S_limited)
    f_subbed = [dH.subs(cD),dS.subs(cD),dN.subs(cD),dC.subs(cD)]
    fps = sp.solve(f_subbed,[H, S, N, C])
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
    state= [funcs[-2]<1/(1+cD[eps]),  funcs[-1]<1/(1+cD[eps])]

    F = sp.Matrix(endo_symbolic(state[0],state[1])).subs(cD)
    J = F.jacobian([H,S,N,C])
    J = J.subs([(H,yStar[0]), (S,yStar[1]), (N,yStar[2]), (C,yStar[3])])

    eig, _ = la.eig(np.array(J.tolist(), dtype=float))
    stab, numDir = True, 0

    for val in eig:
        if sp.re(sp.N(val)) > 0:
            stab = False
            numDir += 1

    return stab #, numDir


def check_illegal(state,yStar,cD):
    muH, muS, rhoDOC, rhoPhoto, pS, rH, rS, netNH, netNS, eH, eS = make_funcs(0,yStar,cD)
    currState = [bool(eH<(1/(cD[eps]+1))) , bool(eS<(1/(cD[eps]+1)))]
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

    if not ignore_H_lim:                          
        fps = []
        for i, j in [(0,0), (0,1), (1,1), (1,0)]:
            fps = fps + _curate_fps(check_for_fps(i,j,cD), [i,j], cD)
        return fps #OBS returned unorded!
    
    fps = [None]*4
    S_not_N_lim = _curate_fps(check_for_fps(0,0,cD), [0,0], cD)
    S_N_lim     = _curate_fps(check_for_fps(0,1,cD), [0,1], cD)

    if len(S_not_N_lim)+len(S_N_lim)>4:           # Sanity check. At most 4 fixed points should exist when H is not nitrogen limited
        raise ValueError("More than 4 feasible fixed points found. Investigate ODE and parametrization!")

    i = 0
    for fp in S_not_N_lim:
        if fp[1] == 0.0:
            fps[0] = fp
        else:
            fps[1+i] = fp
            i += 1

    i = 0
    for fp in S_N_lim:
        if fp[1] == 0.0:
            fps[0] = fp
        else:
            fps[2+i] = fp
            i += 1

    return fps


def make_symb_bifur(para,span,cons=[],ignore_H_lim=True): 
    pValues = np.linspace(span[0],span[1],30)   
    fixList = []
    try:
        for p in tqdm(pValues,desc=f"Progress for parameter: {para}, in range: {span}", unit="run"):
            new_cons = cons + [(para,p)]
            cD = make_cons(new_cons)
            curated_fps = find_all_fps(new_cons, ignore_H_lim)
            for i, fp in enumerate(curated_fps):
                if fp != None:
                    stab = check_symb_stab(fp, cD)
                    fixList.append([para] + [p] + list(fp) + [i] + [stab] )
                    tqdm.write(f"{para}: {p}, H: {round(fp[0],20)}, S: {round(fp[1],20)}, N: {round(fp[2],20)}, C: {round(fp[3],20)}, fp: {i}, stable = {stab}")
                    #sys.stdout.flush()
    except KeyboardInterrupt:
        tqdm.write("\n>>>> Interupted by user")
    return fixList


##### Bifurcation diagrams by using automatic differentiation and numerical root finding #############  
def find_fp_num(func, x0):
    sol = opt.root(func, x0, method="hybr", tol=1e-12, options={"xtol":1e-12,"maxfev":0,"eps":0.1})
    if sol.success:
        return sol.x
    else:
        return None


def check_aut_stab(func, ystar):
    jac = jax.jacfwd(func)
    eigenvals, _ = la.eig(jac(ystar), check_finite=False)
    for eig in eigenvals:
        if eig.real>0: return False
    return True


def make_num_bifur(para, span, base_cons=[]):
    """Creates a numerically solved bifuraction diagram starting from a the standard symbolic solution for fixed points. Used to compare results to smooth approxmations
    
    Args:
        para (sp.Symbol): The sympy symbol of the parameter 
        span (list): 2D list choosing parameter span of bifurcation diagram
        base_cons (list): List of base parameter values
    
    Returns:
        None
    """
    standard_para_value = make_cons(base_cons)[para]
    step = (span[1]-span[0])/150

    symb_fps = find_all_fps(base_cons)
    data = []
    for fp_num, standard_fp in enumerate(symb_fps):
        for k in range(2):
            para_list = np.arange(standard_para_value, span[k], (-1)**(k+1)*step)
            fp_old = np.array(standard_fp, dtype="float64")
            for para_value in para_list:
                cD = make_cons(base_cons+[(para,para_value)])
                func = lambda y: endo(0,y,cD,min_func=min_approx)

                fp = find_fp_num(func, fp_old)
                if fp is not None:
                    stab = check_aut_stab(func, fp)
                    data.append([para_value, fp[0], f"H{fp_num}", stab])
                    data.append([para_value, fp[1], f"S{fp_num}", stab])
                    fp_old = fp

    df = pd.DataFrame(data, columns=["para_value", "value", "point_type", "stability"])
    palette = sns.color_palette("Blues", n_colors=4) + sns.color_palette("Greens", n_colors=4)
    g = sns.scatterplot(df, x="para_value", y="value", hue="point_type", hue_order=["H0", "H1", "H2", "H3", "S0", "S1", "S2", "S3"], palette=palette,  
                    style="stability", style_order=[True, False])
    g.set_ylabel(info.loc["H", "unit"])
    g.set_xlabel(f"${para.name}$")
    plt.show()



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
    columns = info["tex_symbol"].to_list()

    data = np.r_[y, [y[1,:]/y[0,:]], funcs ]
    df = pd.DataFrame(np.transpose(data), index=t, columns=columns)

    if save_name:
        df.to_csv("sims/sim_" + save_name)

    return df


def save_bifur_data(paras=None, cons=[], save_name=""):
    para_spans = { eps: [1,1.4], to: [1,1.25], pmax: [0.01,1.0], uSmax: [0.005,0.09], uHmax: [0.0,0.01], KNS: [0.0001,0.12], KNH: [0.0,0.002], 
            delC: [0.0,0.7], CI: [0.0,0.15], delN: [0.0,0.5], NI: [0.0,0.002], mH: [0.01,0.05], mS: [0.03,0.2], KCO2: [0.0001,0.06], rho0: [0,0.1],
            QFood: [0,0.2], QS: [0.01,0.2], QH: [0.01,0.2], g: [0.01, 1], iota: [0,0.1] }
    subset = dict( [(p, para_spans[p]) for p in paras]) if paras else para_spans

    data = []
    for p, span in subset.items():
        data = data + make_symb_bifur(p, span, cons, ignore_H_lim=True)
    
    df = pd.DataFrame(data, columns=["para_name", "para_value", "H", "S", "N", "C", "fp_num", "stable"])
    df_long = df.melt(id_vars=["para_name", "para_value", "fp_num", "stable"], var_name="var_name", value_name="var_value")

    if save_name:
        df_long.to_csv("sims/"+save_name+".csv", index=False)
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
                                                                                
                    for value, name in zip(fp, ["H", "S", "N", "C"]):
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
    np.savetxt("sims/aoa" + name + "_raw.txt", mat)
    
    return mat


###### Plotting tools ########################################################
def _saveable_name(para):
    return para.name.replace("\\","").replace("{","").replace("}","")


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
    palette  = [info.loc[var, "color"] for var in var_list]
    var_list = [info.loc[var, "tex_symbol"] for var in var_list]
    data     = df[var_list]

    if not ax:
        fig, ax = plt.subplots()

    sns.lineplot(data, palette=palette, ax=ax, dashes=False, legend="brief", zorder=1)   # dashes = [(1,0), (1,1), osv.]

    ax.set_yscale(yscale)
    ax.set_ylim(bottom=ybottom, top=ytop)
    #ax.margins(y=0.05)

    return ax


def plot_bifur(path, paras, vars = ["H", "S"], save_name = ""):
    """
    Plots bifurcation diagram for one or more variables for specified parameters. If more parameters are given puts plots together in a comparable figure.
    
    Args:
        path (str): Path to csv dataframe (long format) with bifurcation data.
        paras (list[str]): List of which parameters should be plotted.
        vars (list[str], optional): Which variables fixed points should be displayed in the bifuraction plot. Defualt is H and S
        save_name (str, optional): If non-empty saves figure as png in figs/bifur under save_name.

    Returns:
        sns.FacetGrid: Grid of bifurcation subplots.
    """
    if isinstance(path, str):
        df = pd.read_csv(path)
    else:
        df = path
    
    ## Creating a color mapping
    cmap = {}
    gradient = {"H": "Blues", "S": "Greens", "N":"RdPu", "C":"Greys"}
    for var in vars:
        pallette = sns.color_palette(gradient[var], n_colors=4)
        for fp_num in range(4):
            cmap[(var, fp_num)] = pallette[fp_num]
    
    ## Curating df
    df["combined_hue"] = list(zip(df["var_name"], df["fp_num"]))
    subset = df[(df["para_name"].isin([str(p) for p in paras])) & (df["var_name"].isin(vars))]

    subset.loc[:, "fp_num"] = subset["fp_num"].replace({1:0, 3:2})

    ## Setting up the plot
    num_col = round(len(paras)**(1/2))
    g = sns.relplot(subset, x="para_value", y="var_value", hue="combined_hue", col="para_name", col_wrap=num_col, col_order=[str(p) for p in paras],
                    palette=cmap, kind="line", legend="brief", style="fp_num", style_order=[2,0],
                    size="stable", sizes={True:3, False:1.5}, size_order=[True, False],
                    facet_kws=dict(sharex=False))
    
    g.figure.set_size_inches(6, 5)
    g.figure.tight_layout(rect=[0, 0, 0.80, 1])

    ## Customizing legends
    legend = g._legend
    handles = legend.legend_handles
    labels = ["State", "Symbiotic", "Dysbiotic", "Stability", "Stable", "Unstable"]
    legend.remove()
    new_leg = g.figure.legend(handles=handles[len(vars)*4+1:len(vars)*4+9], labels=labels, loc="center right", frameon=True)
    new_leg.texts[0].set_fontweight("bold")
    new_leg.texts[3].set_fontweight("bold")
    new_leg.texts[0].set_ha("left")
#

    g.set_titles("")        # Set new titles and x labels
    g.set_ylabels(info.loc[vars[0], "unit"])
    for ax, xlabel in zip(g.axes.flat, [p.name for p in paras]):
        ax.set_xlabel(f"${xlabel}$")

    ## Setting subplot names
    for ax, subname in zip(g.axes.flat, "abcdefghijklmnop"):
        ax.text(0.02, 0.98, f"({subname})", transform=ax.transAxes, va="top", ha="left", fontweight = "bold")


    ## Saving 
    if save_name:
        plt.savefig("figs/bifurs/" + save_name + ".png", dpi=300, bbox_inches="tight")
        plt.savefig("figs/pdf_figs/" + save_name + ".pdf", bbox_inches="tight")
    return g


def plot_2D_bifur(path, var, vmax=None, ax=None, save_name="", cbar=False, cmap="plasma", annot=False, red_line=False):
    """
    Plots 2D bifurcation diagram of given dataframe. Dataframe should have been created using make_2D_bifur()-function
    
    Args:
        path (str or pd.DataFrame): Path to load dataframe or actual dataframe
        var (str): Variable whos value are plotted. Sither "H", "S", "N" or "C"
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
    df_fp3 = df[(df["name"]==var) & (df["fp_num"]==3)].pivot(index=cols[0], columns=cols[1], values="value").astype(float)  # non-zero N-lim state
    df_fp1 = df[(df["name"]==var) & (df["fp_num"]==1)].pivot(index=cols[0], columns=cols[1], values="value").astype(float)  # non-zero C-lim state
    df_fp0 = df[(df["name"]==var) & (df["fp_num"]==0) & (df["stability"]==1)].pivot(index=cols[0], columns=cols[1], values="value").astype(float) # Semi trivial fixedpoint that is stable

    if var == "S" and annot:
        annot_df = ( (df_fp3>df_fp1) | (df_fp3.notna() & df_fp1.isna()) ).replace({True:"*", False:""})
        annot_df = annot_df[::-1]
    else:
        annot_df = None
    sub_df = df_fp3.fillna(df_fp0)

    if not ax:
        fig, ax = plt.subplots(1,1,figsize=(5.5, 4),constrained_layout=True)
        cbar = True

    sns.heatmap(sub_df.iloc[::-1], linewidth=0.5, vmin=0, vmax=vmax, annot=annot_df, fmt="", annot_kws={"color":"white"}, 
                                    cbar_kws={"label": info.loc[var, "unit"]}, cmap=cmap, ax=ax, cbar=cbar)  #cmap="magma", "plasma", "cividis" all cbf
    
    if red_line: ax.axhline(y=5, color="red", ls="--")
    
    skip = 3
    xtick_pos = np.arange(0, sub_df.shape[1], skip) + 0.5
    xtick_lab = [f"{x:.2f}" for x in sub_df.columns[::skip]]
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_lab, rotation=45)

    ytick_pos = np.arange(sub_df.shape[0] - 1, -1, -skip)
    ytick_lab = [f"{sub_df.index[-i-1]:.2f}" for i in ytick_pos]
    ax.set_yticks(ytick_pos+0.5)
    ax.set_yticklabels(ytick_lab)

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

   
    label_list = ["H", "S", "N", "C", "S/H"]
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


    label_list = ["H", "S", "N", "C", "S/H"]

    ax.set_xlabel(f"$H$ ({info.loc[label_list[x_ind], "unit"]})")
    ax.set_ylabel(f"$S/H$ ({info.loc[label_list[y_ind], "unit"]})")

    if save_name:
        plt.savefig("figs/plotted_sims/" + save_name + ".png")
    return im



if __name__ == "__main__":  
    make_num_bifur(uSmax,[0.005,0.08])
    make_num_bifur(eps,[1,1.3])
    