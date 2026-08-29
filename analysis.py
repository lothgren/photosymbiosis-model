#######################################################################################################################################################
## # In this script the final simulations and graphics for the publication and oral presentations is created using the model and the analysis tools ###
#######################################################################################################################################################

from analysisTools import *
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches
import matplotlib as mpl


###### Some simulations
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


def big_sim(figsize, large_fs, small_fs, cons=[],phase_legend=True):
    puls_var = eps
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

    sns.set_theme(context="paper", style="ticks")
    fig, axs = plt.subplots(4,2, figsize=figsize, sharex=True, constrained_layout=True)

    for col, df in enumerate([df0,df1]):
        ax0, ax1, ax2, ax3 = axs[:,col]
        twin0 = ax0.twinx()
        twin1 = ax1.twinx()

        plot_sim(df, ["H", "S"], ax0)
        plot_sim(df, ["S/H"], twin0, "linear")
        
        plot_sim(df, ["N"],ax1,  "linear", ybottom=0-1e-3)
        plot_sim(df, ["C"],twin1,"linear", ybottom=0-1e-2)

        plot_sim(df, ["netNH", "netNS"], ax2, "linear")
        
        plot_sim(df, ["muH", "rhoDOC", "rhoPhoto"],  ax3, "linear")


        # plotting straight lines
        ax2.axhline(y=0, color = "k", dashes=(2,2))
        net_N_time = _find_crossing(df["$H$ net N upt."])

        abc = "abcdefgh"
        titles = ["Biomass", "Nutrient pools", "Per-capita net N uptake", "Per-capita host C assimilation"]
        for i, ax in enumerate([ax0, ax1, ax2, ax3]):
            xlim = ax.get_xlim()
            ax.autoscale(enable=False, axis="x")
            ax.axvspan(xlim[0], net_N_time[0], facecolor=p[0], alpha=0.15, zorder=-100)
            ax.axvspan(net_N_time[0], 350,     facecolor=p[3], alpha=0.15, zorder=-100)
            ax.axvspan(350, 350+puls_duration, facecolor=p[4], alpha=0.50, zorder=-100)
            if col==0:
                ax.axvspan(350+puls_duration, xlim[1], facecolor=p[2], alpha=0.15, zorder=-100)
            else:
                ax.axvspan(350+puls_duration, xlim[1], facecolor=p[7], alpha=0.15, zorder=-100)
           
            ax.set_title(titles[i], fontsize = large_fs)
            ax.text(0.02, 0.95, f"({abc[col+2*i]})", transform=ax.transAxes, va="top", ha="left", fontweight = "bold", fontsize=small_fs)  # <--- here I added the labeling for now...

            ax.axvline(x=net_N_time[0], color="k",dashes=(1,1))
        
        #ax1.text(x=350+puls_duration-15, y=0.01, s=r"$\longleftarrow $" + "Heat wave", fontsize = small_fs)
        #if net_N_time:
        #    ax2.text(x=net_N_time[0]+5, y =-0.004, s=r"$\leftarrow$ Switch in H's N-uptake", fontsize = small_fs)

        ## Setting ylabels and xlabels
        if col == 0:
            for ax, name in zip([ax0,ax1,ax2,ax3], ["H", "N", "netNH", "muH"]):
                ax.set_ylabel(info.loc[name, "unit"], fontsize=large_fs)
            twin0.tick_params(axis="y", which="both", right=False, labelright=False)
            twin1.tick_params(axis="y", which="both", right=False, labelright=False)

            h0, l0 = ax0.get_legend_handles_labels()         # Positioning and merging legends
            twin0_h, twin0_l = twin0.get_legend_handles_labels()
            twin0.legend(h0+twin0_h, l0+twin0_l, loc="lower center", fontsize = small_fs, ncols=3)

            h1, l1 = ax1.get_legend_handles_labels()       
            twin1_h, twin1_l = twin1.get_legend_handles_labels()
            ax1.legend(h1+twin1_h, l1+twin1_l, loc="upper right", fontsize = small_fs, ncols=1)

            ax0.get_legend().remove()
            ax2.legend(loc="lower right",  fontsize = small_fs)
            ax3.legend(loc="center right", fontsize = small_fs)
        else:
            for ax in [ax0, ax1, ax2, ax3]:
                ax.tick_params(axis="y", which="both", left=False, labelleft=False)
                ax.get_legend().remove()
            twin0.set_ylabel(info.loc["S/H", "unit"], fontsize=large_fs)
            twin1.set_ylabel(info.loc["C", "unit"],   fontsize=large_fs)
            
            twin0.get_legend().remove()

        twin1.get_legend().remove()

        for ax in [ax0,ax1,ax2]:
            ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        fig.align_ylabels(axs[:,col])

    all_axs = fig.axes
    ## Rescaling col 2
    for i in range(4):
        all_axs[1+2*i].set_ylim(all_axs[2*i].get_ylim())
    for i in range(2):
        all_axs[10+i].set_ylim(all_axs[8+i].get_ylim())

    ## Enforce clip on for twin axes
    for line in twin0.lines:
        line.set_clip_on(True)
        line.set_clip_path(ax0.patch)

    ## Titles and x-labels
    all_axs[0].text(0.5, 1.5, "Low nutrient availability",  transform=all_axs[0].transAxes, va="top", ha="center", fontweight = "bold", fontsize=large_fs)
    all_axs[1].text(0.5, 1.5, "High nutrient availability", transform=all_axs[1].transAxes, va="top", ha="center", fontweight = "bold", fontsize=large_fs)
    all_axs[6].set_xlabel("d", fontsize=large_fs)
    all_axs[7].set_xlabel("d", fontsize=large_fs)

    ## Making background color legend
    infect    = mpatches.Patch(color=p[0], alpha=0.15, label="Infection")
    switch    = mpatches.Patch(color=p[3], alpha=0.15, label="Switch in N-comp.")
    heat_wave = mpatches.Patch(color=p[4], alpha=0.5, label="Heat wave")
    go_back   = mpatches.Patch(color=p[2], alpha=0.15, label="Return to symbiotic state")
    go_away   = mpatches.Patch(color=p[7], alpha=0.15, label="Switch to dysbiotic state")
    handles = [infect, switch, heat_wave, go_back, go_away]
    if phase_legend:
        fig.legend(handles=handles, ncol=3, loc="outside lower center", frameon=False,
                            borderaxespad=0.0, title="Phase", fontsize=small_fs, title_fontproperties={'weight': 'bold', "size":small_fs},
                            handlelength=0.7, handleheight=0.7, columnspacing=1.5, handletextpad=0.6)
    return fig, axs


def suppl_estab():
    t,y,funcs,cD = multEvents([5,0.001,0.019,0.15], [0,300], cons=[], show_start=True, eventList=[])
    df = make_df(t,y,funcs)

    fig, axs = plt.subplots(2, 2, figsize=(6,3), constrained_layout=True)
    sns.set_theme(context="paper", style="ticks")
    large_fs = 10
    small_fs = 8


    plot_sim(df, ["H", "S"], axs[0,0])
    twin00 = axs[0,0].twinx()
    plot_sim(df, ["S/H"], twin00, "linear", ytop=0.8)
    
    twin10 = axs[1,0].twinx()
    plot_sim(df, ["N"], axs[1,0], "linear", ybottom=0-1e-3)
    plot_sim(df, ["C"], twin10,   "linear", ybottom=0-1e-2)

    plot_sim(df, ["netNH", "netNS"], axs[0,1], "linear")
    plot_sim(df, ["muH", "rhoDOC", "rhoPhoto"],  axs[1,1], "linear")

    # Straight lines
    axs[0,1].axhline(y=0, color = "k", dashes=(2,2))

    # Legends
    h0, l0 = axs[0,0].get_legend_handles_labels()        
    twin0_h, twin0_l = twin00.get_legend_handles_labels()
    axs[0,0].legend(h0+twin0_h, l0+twin0_l, loc="lower right", fontsize = small_fs, ncols=1)
    twin00.legend().remove()

    h1, l1 = axs[1,0].get_legend_handles_labels()        
    twin1_h, twin1_l = twin10.get_legend_handles_labels()
    axs[1,0].legend(h1+twin1_h, l1+twin1_l, loc="center right", fontsize = small_fs, ncols=1)
    twin10.legend().remove()

    axs[0,1].legend(loc="center right",  fontsize = small_fs)
    axs[1,1].legend(loc="lower right",  fontsize = small_fs)

    # Titles
    axs[0,0].set_title("Biomass", fontsize = large_fs)
    axs[1,0].set_title("Nutrient pools", fontsize = large_fs)
    axs[0,1].set_title("Per-capita net N uptake", fontsize = large_fs)
    axs[1,1].set_title("Per-capita host C assimilation", fontsize = large_fs)
    
    # Tiks labels
    axs[0,0].set_ylabel(info.loc["H", "unit"], fontsize=large_fs)
    twin00.set_ylabel(info.loc["S/H", "unit"], fontsize=large_fs)
    axs[1,0].set_ylabel(info.loc["N", "unit"], fontsize=large_fs)
    twin10.set_ylabel(info.loc["C", "unit"],   fontsize=large_fs)
    axs[0,1].set_ylabel(info.loc["netNH", "unit"], fontsize=large_fs)
    axs[1,1].set_ylabel(info.loc["muH", "unit"],   fontsize=large_fs)

    axs[0,0].tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    axs[0,1].tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    axs[1,0].set_xlabel("d", fontsize=large_fs)
    axs[1,1].set_xlabel("d", fontsize=large_fs)

    ctx = sns.plotting_context("paper")
    twin00.tick_params(axis="both", labelsize=ctx["axes.labelsize"])
    twin10.tick_params(axis="both", labelsize=ctx["axes.labelsize"])




    # Sublabels
    for ax, sub_label in zip(axs.ravel(), "abcd"):
        ax.text(0.02, 0.95, f"({sub_label})", transform=ax.transAxes, va="top", ha="left", fontweight = "bold", fontsize=small_fs)

        #xlim = ax.get_xlim()
        #ax.autoscale(enable=False, axis="x")
        #ax.axvspan(xlim[0], xlim[1], facecolor=p[7], alpha=0.15, zorder=-100)


def suppl_zoom_in(large_fs, small_fs):
    fig, axs = big_sim(figsize=(6,6), large_fs=large_fs, small_fs=small_fs)

    axs[0,0].set_xlim(320, 385+30)
    for i in range(2):
        axs[0,i].set_ylim(10-0.5,300)
        axs[1,0].set_ylim(None, 0.01)

    axs[1,0].get_legend().set_loc("center left")
    axs[2,0].legend(loc="lower left",  fontsize = small_fs)
    axs[3,0].legend(loc="center left",  fontsize = small_fs)




def big_aoa_plot(figsize, large_fs, small_fs):
    fig, axs = plt.subplots(3,3, figsize=figsize, sharex=True, sharey=True, constrained_layout=True)

    for i, NI_val in enumerate([9e-05, 0.00027, 0.00045]):
        for j, eps_val in enumerate([1.0, 1.1, 1.2]):
            im = plot_aoa(f"sims/aoa_N_I={NI_val}_epsilon={eps_val}_raw.txt", [0,-1], ax=axs[i,j])
            fps = find_all_fps([(NI,NI_val), (eps,eps_val)])
            if fps[3]:
                line, = axs[i,j].plot(fps[3][0],fps[3][1]/fps[3][0], marker="x", ms=5, color="white", markeredgewidth=1.5, zorder=10)
                line.set_path_effects([pe.Stroke(linewidth=4, foreground="black"), pe.Normal()])
            if i == 0: axs[i,j].text(0.5, 1.1, r"$\epsilon$ "+f"= {eps_val}", rotation=0, ha="center", va="top", transform=axs[i, j].transAxes, fontsize=large_fs)
            if j == 2: axs[i,j].text(1, 0.5, f"$N_I$ = {NI_val}", rotation=-90, ha="left", va="center", transform=axs[i, j].transAxes, fontsize=large_fs)
            axs[i,j].set_xlabel("")
            axs[i,j].set_ylabel("")
            if i in [0,1]:axs[i,j].xaxis.set_visible(False)
            if j in [1,2]:axs[i,j].yaxis.set_visible(False)
    fig.supylabel("Symbiont load ($S/H$)", fontsize=large_fs)
    fig.supxlabel("Host biomass ($H$)", fontsize=large_fs)
    cbar = fig.colorbar(im, ax=axs, label="Probability to enter symbiotic state", orientation="vertical", fraction=0.05, pad=0.04)
    cbar.ax.tick_params(labelsize=large_fs)


def suppl_aoa_plot(figsize, large_fs, small_fs):
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(1, 4, width_ratios=[1,1,1,0.08], wspace=0.05)
    axs = [fig.add_subplot(gs[0,i]) for i in range(3)]
    cax = fig.add_subplot(gs[0,3])

    for j, eps_val in enumerate([1.0,1.1,1.2]):
        im = plot_aoa(f"sims/aoa_u_S,max=0.0325_epsilon={eps_val}_raw.txt", [0,-1], ax=axs[j])

        fps = find_all_fps([(uSmax,0.0325), (eps,eps_val)])
        if fps[3]:
            line, = axs[j].plot(fps[3][0],fps[3][1]/fps[3][0], marker="x", ms=5, color="white", markeredgewidth=1.5, zorder=10)
            line.set_path_effects([pe.Stroke(linewidth=4, foreground="black"), pe.Normal()])
        axs[j].text(0.5, 1.14, r"$\epsilon$ = "+ f"{eps_val}", rotation=0, ha="center", va="top", transform=axs[j].transAxes, fontsize=large_fs)
        axs[j].set_xlabel("")
        if j in [1,2]: axs[j].yaxis.set_visible(False)

    axs[0].set_ylabel("")
    fig.supylabel("Symbiont load ($S/H$)", fontsize=large_fs)
    fig.supxlabel("Host biomass ($H$)", fontsize=large_fs)
    axs[j].text(1, 0.5, r"$u_{S,\max}=0.0325$", rotation=-90, ha="left", va="center", transform=axs[j].transAxes, fontsize=large_fs)
    cbar = fig.colorbar(im, cax=cax, label="Probability", orientation="vertical")


def big_2D_plot(figsize, large_fs, small_fs):
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = gspec.GridSpec(nrows=2, ncols=3, width_ratios=[1, 1, 0.05], wspace=0.05, hspace=0.05, figure=fig)
    axs = np.empty((2,2), dtype=object)
    cbars = np.empty(2, dtype=object)

    for i, cmap, var in zip([0, 1], ["plasma", "viridis"], ["H", "S"]):
        axs[i,0] = fig.add_subplot(gs[i,0])
        axs[i,1] = fig.add_subplot(gs[i,1])
        cax = fig.add_subplot(gs[i,2])

        df = pd.concat([pd.read_csv("sims/2D_bifur_eps_uSmax.csv"), pd.read_csv("sims/2D_bifur_eps_pmax.csv")])   # Finding max value to scale colorbar
        vmax = df.loc[(df["name"] == var) & (df["fp_num"] == 3), "value"].max()                                                       # 
        hm0 = plot_2D_bifur(path="sims/2D_bifur_eps_uSmax.csv", var=var, ax=axs[i,0], vmax=vmax, cmap=cmap, red_line=True)
        hm1 = plot_2D_bifur(path="sims/2D_bifur_eps_pmax.csv",  var=var, ax=axs[i,1], vmax=vmax, cmap=cmap, red_line=True)
        cbar = fig.colorbar(hm1.collections[0], cax=cax)
        cbars[i] = cbar

    cbars[0].set_label("Host biomass ($H$)", fontsize=large_fs) 
    cbars[1].set_label("Symbiont biomass ($S$)", fontsize=large_fs) 

    axs[0,0].set_xlabel("")
    axs[0,1].set_xlabel("")
    axs[1,0].set_xlabel(r"Max N uptake ($u_{S,\max}$)", fontsize=large_fs)
    axs[1,1].set_xlabel(r"Max C fixation ($p_{\max}$)", fontsize=large_fs)
    
    axs[0,0].set_ylabel(r"Metabolic costs ($\epsilon$)", fontsize=large_fs)
    axs[1,0].set_ylabel(r"Metabolic costs ($\epsilon$)", fontsize=large_fs)
    axs[0,1].set_ylabel("")
    axs[1,1].set_ylabel("")

    for ax, subname in zip(axs.ravel(), "abcd"):
        ax.text(0.02, 1, f"({subname})", transform=ax.transAxes, va="top", ha="left", fontweight = "bold")
    
    return fig, axs, cbars


#### Creating data and plots ##############################################################
def make_data():
    ## All 1D bifurcations
    save_bifur_data(save_name="bifur_df")
    ## Area of attraction plots
    for NI_val in [9e-05, 27e-05, 45e-05]:
        for eps_val in [1.0, 1.1, 1.2]:
            prob_of_states(n=30000, cons=[(NI,NI_val), (eps,eps_val)])
    
    for eps_val in [1.0, 1.1, 1.2]:
        prob_of_states(n=30000, cons=[(uSmax, 0.0325), (eps,eps_val)])
        prob_of_states(n=30000, cons=[(uSmax, 0.05), (eps,eps_val)])


    ## 2D bifurcations
    make_2D_bifur(eps,uSmax,[1,1.5],[0.001,0.07], 15, save_name="eps_uSmax")
    make_2D_bifur(eps,pmax,[1,1.5],[0.001,1], 15, save_name="eps_pmax")
    
    make_2D_bifur(pmax,uSmax,[0.01,1],[0.001,0.07], 15, save_name="pmax_uSmax")
    make_2D_bifur(eps,NI,[1,1.5],[0.0,0.00175], 15, save_name="eps_NI")
    make_2D_bifur(eps,CI,[1,1.5],[0.07,0.15], 15, save_name="eps_CI")

    make_2D_bifur(eps,rho0,[1,1.5],[0.001,0.1], 15, save_name="eps_rhoFood")
    make_2D_bifur(eps,QFood,[1,1.5],[0.01,0.20], 15, save_name="eps_QFood")


def plot_and_save():
    large_fs = 10
    small_fs = 8

    ## Plot large simulation displaying estab
    big_sim(figsize=(6,6), large_fs=large_fs, small_fs=small_fs, phase_legend=False)
    #plt.savefig("figs/plotted_sims/estab_plus_heat_wave.png", dpi=300, bbox_inches="tight") 
    plt.savefig("figs/pdf_figs/estab_plus_heat_wave.pdf", bbox_inches="tight", transparent=False)
    #plt.savefig("submission/figure2.eps", format="eps", bbox_inches="tight")

    ## Plot area of attraction
    big_aoa_plot(figsize=(6,5), large_fs=large_fs, small_fs=small_fs)
    #plt.savefig("figs/plotted_sims/aoa_collection.png", dpi=300, bbox_inches="tight")
    plt.savefig("figs/pdf_figs/aoa_collection.pdf", bbox_inches="tight")
    #plt.savefig("submission/aoa_collection.tiff", bbox_inches="tight")

    ## Plot 2D bifurcations
    big_2D_plot((6,5), large_fs=large_fs, small_fs=small_fs)                               
    #plt.savefig("figs/plotted_sims/2D_bifur.png", dpi=300, bbox_inches="tight")
    plt.savefig("figs/pdf_figs/2D_bifur.pdf", bbox_inches="tight")
    #plt.savefig("submission/figure3.eps", format="eps", bbox_inches="tight")

    ## Supplementary plots
    plot_bifur("sims/bifur_df.csv", [eps, rho0, NI, CI ],       save_name="bifur_external")
    plot_bifur("sims/bifur_df.csv", [uSmax, KNS, pmax, KCO2], save_name="bifur_symbiont")
    plot_bifur("sims/bifur_df.csv", [QFood,QS,QH, mS], save_name="bifur_quotas")
    

    suppl_aoa_plot((6,5/3), large_fs=large_fs, small_fs=small_fs)
    #plt.savefig("figs/plotted_sims/suppl_aoa.png", dpi=300, bbox_inches="tight")
    plt.savefig("figs/pdf_figs/suppl_aoa.pdf", bbox_inches="tight")
    #plt.savefig("submission/suppl_aoa.tiff", bbox_inches="tight")

    suppl_estab()
    #plt.savefig("figs/plotted_sims/suppl_estab.png", dpi=300, bbox_inches="tight")
    plt.savefig("figs/pdf_figs/suppl_estab.pdf", bbox_inches="tight")
    #plt.savefig("submission/suppl_estab.tiff", bbox_inches="tight")

    suppl_zoom_in(small_fs=small_fs, large_fs=large_fs)
    plt.savefig("figs/pdf_figs/suppl_zoom_in.pdf", bbox_inches="tight")
    
   

if __name__ == "__main__":
    #make_data()
    plot_and_save()
    #plot_slide_pics()

    plt.show()