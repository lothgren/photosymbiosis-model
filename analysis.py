#This script conatins all 

from analysisTools import *
import sympy as sp

###### Some simulations
def mult_events(y0, event_time=[], event=[], base_cons=[], show_start = True): # under construction
    if not show_start:
        _, y, _, _ = sim_system(y0, [0, 1000], cons=base_cons)
        y0 = y[:,-1]

    t, y, funcs = np.empty((1,1)), np.empty((4,1)), np.empty((13,1))
    t_start = 0
    for i, t_end in enumerate(event_time):
        if event[i]:
            ny0, new_cons = event[i]
        else:
            ny0, new_cons = y0, base_cons
        
        for i in range(len(y0)):
            if not isinstance(ny0[i], (int, float)):
                ny0[i] = y0[i]

        newt, newy, new_funcs, new_cD = sim_system(ny0, [t_start, t_end], cons=base_cons + new_cons)
        newy[1,newy[1,:]==0] = np.nan
        
        t = np.append(t,newt)
        y = np.c_[y, newy]
        funcs = np.c_[funcs, new_funcs]

        t_start, y0 = t_end, newy[:,-1]
    print(y)
    return t, y, funcs


def multEvents(y0,tSpan,cons=[],eventList=[], show_start = True):
    t, y, funcs, cD = sim_system(y0,tSpan,cons)
    y[1,y[1,:]==0] = np.nan
    y0Old, spanOld = y[:,-1], tSpan

    if not show_start:
        t = np.empty((1,1))
        y, funcs = np.empty((4,1)), np.empty((11,1))
        spanOld = [0, 0]

    for event in eventList:
        ny0, newSpan, newCons = event
        
        for i in range(len(y0)):
            if ny0[i] == None:
                ny0[i] = y0Old[i]
        
        if not isinstance(newSpan,list):
            newSpan = [spanOld[1],spanOld[1]+newSpan]

        t2, y2, funcs2, cD2 = sim_system(ny0, newSpan, cons + newCons)
        y2[1,y2[1,:]==0] = np.nan

        t = np.append(t,t2)
        y = np.c_[y, y2]
        funcs = np.c_[funcs, funcs2]
        y0Old, spanOld = y2[:,-1], newSpan
    return t,y,funcs,cD


def _find_crossing(df):
    ind_list = []
    for i in range(len(df)-1):
        if df.iloc[i] < 0 and df.iloc[i+1] > 0:
            ind_list.append(df.index[i])
    return ind_list


def big_sim(cons=[]):
    puls_var = s
    puls_size = 1.5
    puls_duration = 35

    # Establishment under normal circumstances + s-puls
    t,y,funcs,cD = multEvents([25,0,0.001,0.001], [0,200], cons=cons, show_start=False, eventList=[
        [[None, 0,   None, None],             50,       [] ],
        [[None, 0.1, None, None],            300,       [] ],
        [[None, None, None, None], puls_duration, [(puls_var,puls_size)]],
        [[None, None, None, None],            200,      [] ]
    ])

    #Establishing during highten N_I = s-puls
    y0 = y[:,-1]
    spec_var = NI
    spec_val = 0.00045
    t1,y1,funcs1,cD1 = multEvents([25,0,0.001,0.001], [0,200], cons=cons+[(spec_var,spec_val)], show_start=False, eventList=[
        [[None, 0,   None, None],             50,       [] ],
        [[None, 0.1, None, None],            300,       [] ],
        [[None, None, None, None], puls_duration, [(puls_var,puls_size)] ],
        [[None, None, None, None],           200,       [] ]
    ])
    
    df0 = make_df(t,y,funcs)
    df1 = make_df(t1,y1,funcs1)

    fig, axs = plt.subplots(4,2, figsize=(9.6,7.2))
    inner_fs = "small"
    outer_fs = "medium"

    for col, df in [(0, df0), (1, df1)]:
        ax0, ax1, ax2, ax3 = axs[:,col]
        twin0 = ax0.twinx()
        twin1 = ax1.twinx()

        plot_sim(df, ["H", "E"], ax0)
        plot_sim(df, ["E/H"], twin0, "linear")
        
        plot_sim(df, ["N"],ax1,  "linear", ybottom=0-1e-3)
        plot_sim(df, ["C"],twin1,"linear", ybottom=0-1e-2)

        plot_sim(df, ["netNH", "netNE"], ax2, "linear")
        
        plot_sim(df, ["muH", "rhoDOC", "rhoPhoto"],  ax3, "linear")


        # plotting straight lines
        ax2.axhline(y=0, color = "k", dashes=(2,2))
        net_N_time = _find_crossing(df["net $N$ uptake by $H$"])

        abc = "abcdefgh"
        titles = ["Population density", "Inorganic pools", "Net DIN uptake", "Host carbon assimialtion and growth"]
        for i, ax in enumerate([ax0, ax1, ax2, ax3]):
            color = "k"
            xlim, ylim = ax.get_xlim(), ax.get_ylim()
            ax.add_artist(plt.Rectangle((xlim[0],ylim[0]), abs(xlim[0])+350, abs(ylim[0])+ylim[1], facecolor="g", alpha=0.1, zorder=-100))
            if col == 0:
                ax.add_artist(plt.Rectangle((350+puls_duration,ylim[0]), abs(xlim[1])+350+puls_duration, abs(ylim[0])+ylim[1], facecolor="g", alpha=0.1, zorder=-100))
            else:
                ax.add_artist(plt.Rectangle((350+puls_duration,ylim[0]), abs(xlim[1])+350+puls_duration, abs(ylim[0])+ylim[1], facecolor="r", alpha=0.1, zorder=-100))
            ax.axvspan(350, 350+puls_duration, facecolor="r", alpha=0.3, zorder=-100)

            ax.set_title(titles[i], fontsize = outer_fs)
            ax.text(0.02, 0.95, f"({abc[col+2*i]})", transform=ax.transAxes, va="top", ha="left", fontweight = "bold")  # <--- here I added the labeling for now...

            ax.axvline(x=net_N_time[0], color="k",dashes=(1,1))
        
        ax1.text(x=350+puls_duration-15, y=0.01, s=r"$\longleftarrow $" + "Heat wave", fontsize = inner_fs)
        if net_N_time:
            ax2.text(x=net_N_time[0]+5, y =-0.004, s=r"$\leftarrow$ Switch in H's N-uptake", fontsize = inner_fs)

        # Setting ylabels
        if col == 0:
            ax0.set_ylabel(r"mol C$_\text{O}$/m$^2$", fontsize=outer_fs)
            ax1.set_ylabel(r"mol DIN/mol C$_\text{O}$", fontsize=outer_fs)
            ax2.set_ylabel("days$^{-1}$")
            ax3.set_ylabel("days$^{-1}$")
            #twin0.set_yticklabels([])
            #twin1.set_yticklabels([])
        else:
            #ax0.set_yticklabels([])
            #ax1.set_yticklabels([])
            #ax2.set_yticklabels([])
            #ax3.set_yticklabels([])
            twin0.set_ylabel(r"mol C$_\text{O}$/mol C$_\text{O}$", fontsize=outer_fs)
            twin1.set_ylabel(r"mol DIC/mol C$_\text{O}$", fontsize=outer_fs)

        # Positioning legends
        legend_fs = "x-small"
        ax0.legend(  loc = "lower right", fontsize = legend_fs)
        twin0.legend(loc = "upper right", fontsize = legend_fs)
        ax1.legend(  loc="center right",   fontsize = legend_fs)
        twin1.legend(loc = "upper right", fontsize = legend_fs)
        ax2.legend(  loc="lower right",   fontsize = legend_fs)
        ax3.legend(  loc="lower right",   fontsize = legend_fs)

    # Rescaling col 2
    all_axs = fig.axes
    #print(all_axs)
    for i in range(4):
        all_axs[1+2*i].set_ylim(all_axs[2*i].get_ylim())
    for i in range(2):
        all_axs[10+i].set_ylim(all_axs[8+i].get_ylim())


    #Titles and x-labels

    all_axs[0].text(0.5, 1.5, "Standard environmental nutrients, " + r"N$_{\boldsymbol{\text{I}}}$ = 90$\cdot$10$^{\boldsymbol{-6}}$", transform=all_axs[0].transAxes, va="top", ha="center", fontweight = "bold")
    all_axs[1].text(0.5, 1.5, "Heightened environmental nutrients, " + r"N$_{\boldsymbol{\text{I}}}$ = 450$\cdot$10$^{\boldsymbol{-6}}$",   transform=all_axs[1].transAxes, va="top", ha="center", fontweight = "bold")
    all_axs[6].set_xlabel("days")
    all_axs[7].set_xlabel("days")

    plt.tight_layout()
    #plt.show()


###### bifurcation diagram
def plotBifur(para,bList,ax=None, save=False, EtoH = False):      ### Already looking good but we can make it prettier with sns
    if ax == None:
        fig, ax = plt.subplots(1,1)

    mList = ["^", "d", "o", "s"] + ["x"]*10
    mfcList = ["none", None]

    ms, alpha = 9.0, 0.7

    for i in range(max(bList[:,-2])+1):
        for j in range(2):
            m, mfc = mList[i], mfcList[j]
            rowInd = (bList[:,-1]==j) & (bList[:,-2]==i)
            if not EtoH:
                ax.plot(bList[rowInd,1],bList[rowInd,3], color = "C2", marker = m, mfc = mfc, ms= ms, alpha=alpha, ls="-")
                ax.plot(bList[rowInd,1],bList[rowInd,2], color = "C0", marker = m, mfc = mfc, ms= ms, alpha=alpha, ls="-")
            else:
                ax.plot(bList[rowInd,1],bList[rowInd,3]/bList[rowInd,2], color = "C1", marker = m, mfc = mfc, ms= ms, alpha=alpha, ls="")

    #ax.legend()
    ax.set_xlabel(f"${para.name}$")
    ylabel = "$E^*/H^*$ (molar ratio)" if EtoH else r"$H^*,\,E^*$ mol C/m$^2$"
    ax.set_ylabel(ylabel)
    if save:
        saveName = para.name.replace("\\","")
        plt.savefig("figs/bifurs/bifur_" + saveName + ".png")


def runCollection(pList,save=False,preView=True,cons=[]):
    bifurList = { s: [1,2], to: [1,1.6], pmax: [0.01,1.0], uEmax: [0.005,0.07], uHmax: [0.0,0.01], KNE: [0.0001,0.12], KNH: [0.0,0.01], 
                 delC: [0.0,0.7], CI: [0.0,0.15], delN: [0.0,0.5], NI: [0.0,0.004], mH: [0.01,0.05], mE: [0.03,0.2], KCO2: [0.0001,0.06], rho0: [0,0.1],
                 QFood: [0,0.2], QE: [0.01,0.2], QH: [0.01,0.2], b: [0.01, 1], eps: [0,0.1] }
    fig, axs = plt.subplots(len(pList), len(pList[0]))

    for i in range(len(pList)):
        for j in range(len(pList[i])):
            para, span = pList[i][j], bifurList[pList[i][j]]
            b_list = np.array(makeSymbBifur(para, span, cons,ignore_H_lim=True))

            if preView:
                if np.ndim(np.squeeze(pList)) >= 2:
                    ax = axs[i,j] 
                elif np.ndim(np.squeeze(pList)) == 1: 
                    ax = axs[i+j]
                else:
                    ax = axs
                plotBifur(para,b_list,ax=ax,save=False)
            
            if save:
                plotBifur(para,b_list,ax=None,save=True)
                plt.close()

    if preView:
        plt.show()


#### Running stuff ##############################################################

def make_data():
    ## All 1D bifurcations
    #save_bifur_data(save_name="bifur_df")

    ## Area of attraction plots
    #prob_of_states(n=30000, cons=[])
    #prob_of_states(n=30000, cons=[(NI,0.00027)])
    #prob_of_states(n=30000, cons=[(s,1.2)])
    #prob_of_states(n=30000, cons=[(uEmax,0.02)])
    #prob_of_states(n=30000, cons=[(uEmax,0.05)])
    #prob_of_states(n=30000, cons=[(NI,0.0009)])

    ## 2D bifurcations
    make_2D_bifur(s,uEmax,[1,1.5],[0.001,0.07], 15, save_name="s_uEmax")
    make_2D_bifur(s,pmax,[1,1.5],[0.001,1], 15, save_name="s_pmax")
    
    make_2D_bifur(pmax,uEmax,[0.01,1],[0.001,0.07], 15, save_name="pmax_uEmax")
    make_2D_bifur(s,NI,[1,1.5],[0.0,0.00175], 15, save_name="s_NI")
    make_2D_bifur(s,CI,[1,1.5],[0.07,0.15], 15, save_name="s_CI")


def big_aoa_plot():
    fig, axs = plt.subplots(3,3, figsize=(9.6,7.2), sharex=True, sharey=True, constrained_layout=True)

    for i, NI_val in enumerate([9e-05, 0.00027, 0.00045]):
        for j, s_val in enumerate([1, 1.1, 1.2]):
            im = plot_aoa(f"sims/aoa_N_I={NI_val}_s={s_val}_raw.txt", [0,-1], ax=axs[i,j])
            fps = find_all_fps([(NI,NI_val), (s,s_val)])
            if fps[3]:
                axs[i,j].plot(fps[3][0],fps[3][1]/fps[3][0],"rx")
            if i == 0: axs[i,j].text(0.5, 1.07, f"$s$ = {s_val}", rotation=0, ha="center", va="top", transform=axs[i, j].transAxes)
            if j == 2: axs[i,j].text(1, 0.5, f"$N_I$ = {NI_val}", rotation=-90, ha="left", va="center", transform=axs[i, j].transAxes)
            if i in [0,1]: axs[i,j].xaxis.set_visible(False)
            if j in [1,2]: axs[i,j].yaxis.set_visible(False)
    #fig.tight_layout()
    cbar = fig.colorbar(im, ax=axs, label="Probalility", orientation="vertical", fraction=0.05, pad=0.04)


def suppl_aoa_plot(figsize):
    fig, axs = plt.subplots(1,3, figsize=figsize, sharex=True, sharey=True, constrained_layout=True)

    for j, s_val in enumerate([1,1.1,1.2]):
        im = plot_aoa(f"sims/aoa_uEmax=0.0325_s={s_val}_raw.txt", [0,-1], ax=axs[j])

        fps = find_all_fps([(uEmax,0.0325), (s,s_val)])
        if fps[3]:
            axs[j].plot(fps[3][0],fps[3][1]/fps[3][0],"rx")

        axs[j].text(0.5, 1.07, f"$s$ = {s_val}", rotation=0, ha="center", va="top", transform=axs[j].transAxes)
        if j in [1,2]: axs[j].yaxis.set_visible(False)
    
    axs[j].text(1, 0.5, r"$u_{E,\max}=0.0325$", rotation=-90, ha="left", va="center", transform=axs[j].transAxes)
    cbar = fig.colorbar(im, ax=axs, label="Probalility", orientation="vertical", fraction=0.05, pad=0.04)


def big_2D_plot():
    fig = plt.figure(figsize=(9.6,7.2))
    gs = gspec.GridSpec(nrows=2, ncols=3, width_ratios=[1, 1, 0.05], wspace=0.3, hspace=0.2)
    axs = np.empty((2,2), dtype=object)

    for i, cmap, var in zip([0, 1], ["plasma", "viridis"], ["H", "E"]):
        axs[i,0] = fig.add_subplot(gs[i,0])
        axs[i,1] = fig.add_subplot(gs[i,1])
        cax = fig.add_subplot(gs[i,2])

        df = pd.concat([pd.read_csv("sims/2D_bifur_s_uEmax.csv"), pd.read_csv("sims/2D_bifur_s_pmax.csv")])   # Finding max value to scale colorbar
        vmax = df.loc[df["name"] == var, "value"].max()                                                       # 
        hm0 = plot_2D_bifur(path="sims/2D_bifur_s_uEmax.csv", var=var, ax=axs[i,0], vmax=vmax, cmap=cmap)
        hm1 = plot_2D_bifur(path="sims/2D_bifur_s_pmax.csv",  var=var, ax=axs[i,1], vmax=vmax, cmap=cmap)
        cbar = fig.colorbar(hm1.collections[0], cax=cax)
        cbar.set_label(f"{var} C-mol/m$^2$")

    axs[0,0].set_xlabel("")
    axs[0,1].set_xlabel("")
    axs[0,1].set_ylabel("")
    axs[1,1].set_ylabel("")

    for ax, subname in zip(axs.ravel(), "abcd"):
        ax.text(0.02, 1, f"({subname})", transform=ax.transAxes, va="top", ha="left", fontweight = "bold")
    fig.tight_layout()



def plot_and_save():
    figsize = (9.6,7.2)  # common figsize for all plots_

    ## Plot large simulation displaying estab
    big_sim()
    plt.savefig("figs/plotted_sims/estab_plus_heat_wave.png", dpi=300, bbox_inches="tight")  # can also save as pdf (optional?). Then dont give dpi arg!

    ## Plot area of attraction
    #big_aoa_plot()
    #plt.savefig("figs/plotted_sims/aoa_collection.png", dpi=300, bbox_inches="tight")


    ## Plot 2D bifurcations
    #big_2D_plot()                               
    #plt.savefig("figs/plotted_sims/2D_bifur.png", dpi=300, bbox_inches="tight")

    ## Supplementary plots
    #plot_bifur("sims/bifur_df", [s, rho0, NI, CI ],       save_name="bifur_external")
    #plot_bifur("sims/bifur_df", [uEmax, KNE, pmax, KCO2], save_name="bifur_symbiont")
    #plot_bifur("sims/bifur_df", [s])

    suppl_aoa_plot((figsize[0],figsize[1]/3))
    plt.savefig("figs/plotted_sims/suppl_aoa.png", dpi=300, bbox_inches="tight")




if __name__ == "__main__":
    #runCollection([[CI, delC], [NI, delN]], cons=[(b,1),(pmax,0.25)])
    #make_data()
    #plot_bifur("sims/bifur_df", [s, uEmax, NI])
    plot_and_save()
    #plot_aoa(f"sims/aoa__u_E,max=0.02_N_I=0.0009_raw.txt", [0,-1])
    
    plt.show()