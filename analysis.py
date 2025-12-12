#Doing all the simulations, last updated 13

from analysisTools import *
import sympy as sp


###### Plotting bifurcation diagrams



###### Some simulations
def mult_events(y0, event_time=[], event=[], base_cons=[], show_start = True): # under construction
    if not show_start:
        _, y, _, _ = simSystem(y0, [0, 1000], cons=base_cons)
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

        newt, newy, new_funcs, new_cD = simSystem(ny0, [t_start, t_end], cons=base_cons + new_cons)
        newy[1,newy[1,:]==0] = np.nan
        
        t = np.append(t,newt)
        y = np.c_[y, newy]
        funcs = np.c_[funcs, new_funcs]

        t_start, y0 = t_end, newy[:,-1]
    print(y)
    return t, y, funcs


def multEvents(y0,tSpan,cons=[],eventList=[], show_start = True):
    t, y, funcs, cD = simSystem(y0,tSpan,cons)
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

        t2, y2, funcs2, cD2 = simSystem(ny0, newSpan, newCons)
        y2[1,y2[1,:]==0] = np.nan

        t = np.append(t,t2)
        y = np.c_[y, y2]
        funcs = np.c_[funcs, funcs2]
        y0Old, spanOld = y2[:,-1], newSpan
    return t,y,funcs,cD


def estabSim():
    _, y_start, _, _ = simSystem([55, 0, 0.05, 0.16], [0,500])
    y0 = y_start[:,-1]   #Establishing under normal conditions
    y0[1] = 1e-1
    tSpan = [0,500]
    t, y, funcs, cD = multEvents(y0,tSpan,cons=[(uEmax,0.035)])
    axs = plot_sim_2(t,y,funcs,cD)


def estab_sim():
    """Newer version of estabSim()"""
    _, y_start, _, _ = simSystem([55, 0, 0.05, 0.16], [0,500])
    y0 = y_start[:,-1]                                          #Setting good initial conditions
    y0[1] = 1e-1
    tSpan = [0,500]

    t, y, funcs, cD = simSystem(y0,tSpan)   #Make the actual simulation
    df = make_df(t, y, funcs, cD)


    scale_fig = 2

    ### State variables
    estab_fig, (ax0, ax1) = plt.subplots(2,1,figsize=(6.4*scale_fig, 4.8*scale_fig))
    twin0 = ax0.twinx()
    twin1 = ax1.twinx()

    plot_sim(df,["H", "E"], ax0)
    plot_sim(df,["E/H"], twin0,"linear")
    
    plot_sim(df, ["N"],ax1,"linear")
    plot_sim(df, ["C"],twin1,"linear")

    ctx = sns.plotting_context("talk")
    fsize = ctx["axes.labelsize"]

    ax1.legend(loc="upper right")
    twin1.legend(loc="right")

    ax0.set_ylabel("mol C/m$^2$", fontsize=fsize)
    twin0.set_ylabel("mol C/mol C", fontsize=fsize)
    ax1.set_ylabel("mol N/mol C", fontsize=fsize)
    twin1.set_ylabel("mol C/mol C", fontsize=fsize)
    ax1.set_xlabel("days", fontsize=fsize)

    estab_fig.savefig("figs/plotted_sims/estab_normal.png",dpi=300, bbox_inches="tight")


    ### Functions of interest
    estab_fig2, (ax2, ax3) = plt.subplots(2,1,figsize=(6.4*scale_fig, 4.8*scale_fig))
    plot_sim(df,["net $N$ uptake by $H$", "net $N$ uptake by $E$"], ax2,"linear")
    plot_sim(df,[r"$\mu_H$", r"$\rho_{food}$", r"$\rho_{photo}$"], ax3,"linear")

    ax2.set_ylabel("days$^{-1}$")
    ax3.set_ylabel("days$^{-1}$")
    ax3.set_xlabel("days")

    estab_fig2.savefig("figs/plotted_sims/estab2_normal.png",dpi=300, bbox_inches="tight")

    plt.show()


def nitro_pulse_sim(s_value):
    _, y_start, _, _ = simSystem([140, 20, 0.003, 0.01], [0,500],cons=[(s,s_value)])
    y0 = y_start[:,-1]   #Establishing under normal conditions
    tSpan = [0,50]
    t, y, funcs, cD = multEvents(y0,tSpan,cons=[(s,s_value)], eventList=[
        [[None, None, 0.03, None],200,[(s,s_value)]]
    ])
    
    df = make_df(t, y, funcs, cD)
    scale_fig = 2

    ### State variables
    fig, (ax0, ax1) = plt.subplots(2,1,figsize=(6.4*scale_fig, 4.8*scale_fig))
    plot_sim(df,["H", "E"], ax0)
    twin0 = ax0.twinx()
    plot_sim(df,["E/H"], twin0,"linear")
    
    plot_sim(df, ["N"],ax1,"linear")
    twin1 = ax1.twinx()
    plot_sim(df, ["C"],twin1,"linear")

    ctx = sns.plotting_context("talk")
    fsize = ctx["axes.labelsize"]

    ax1.legend(loc="upper right")
    twin1.legend(loc="right")

    ax0.set_ylabel("mol C/m$^2$", fontsize=fsize)
    twin0.set_ylabel("mol C/mol C", fontsize=fsize)
    ax1.set_ylabel("mol N/mol C", fontsize=fsize)
    twin1.set_ylabel("mol C/mol C", fontsize=fsize)
    ax1.set_xlabel("days", fontsize=fsize)

    fig.savefig(f"figs/plotted_sims/n_puls_s={s_value}.png",dpi=300, bbox_inches="tight")


def starvation_sim():
    t, y, funcs, cD = multEvents([55, 0.1, 0.05, 0.16],[0,300],cons=[(NI,0.0025)], eventList=[
            [[None, None, None, None],200,[(rho0,0.001),(NI,0.0025)]]
        ])
    plot_sim_2(t,y,funcs,cD)
    #plt.vlines(250, 0, 0.1,"grey")
    #plt.text(250-2,4e-2,"N pulse ->",ha="right")
    #plt.show()


def plot_phase():
    y0 = [55, 0.1, 0.001, 0.016]
    tSpan = [0,500]
    t, y, funcs, cD = multEvents(y0,tSpan,cons=[])

    plt.plot(y[2], y[1]/y[0])

    t, y, funcs, cD = multEvents(y0,tSpan,cons=[(s,1.25)])
    plt.plot(y[2], y[1]/y[0])

    plt.ylabel("E/H")
    plt.xlabel("N")


def _find_crossing(df):
    ind_list = []
    for i in range(len(df)-1):
        if df.iloc[i] < 0 and df.iloc[i+1] > 0:
            ind_list.append(df.index[i])
    return ind_list


def big_sim(cons=[]):
    puls_var = s
    puls_size = 1.3
    puls_duration = 40


    # Establishment under normal circumstances + n-puls
    t,y,funcs,cD = multEvents([25,0,0.001,0.001], [0,200], cons=cons, show_start=False, eventList=[
        [[None, 0,   None, None],             50,       [] ],
        [[None, 0.1, None, None],            300,       [] ],
        [[None, None, None, None], puls_duration, [(puls_var,puls_size)]],
        [[None, None, None, None],            170,      [] ]
    ])

    y0 = y[:,-1]
    spec_var = NI
    spec_val = 0.003
    t1,y1,funcs1,cD1 = multEvents(y0, [0,100], cons=cons+[(spec_var,spec_val)], show_start=True, eventList=[
        [[None, None, None, None],           250,                       [(spec_var,spec_val)] ],
        [[None, None, None, None], puls_duration, [(puls_var,puls_size), (spec_var,spec_val)] ],
        [[None, None, None, None],           150,                       [(spec_var,spec_val)] ]
    ])

    df0 = make_df(t,y,funcs)
    df1 = make_df(t1,y1,funcs1)

    fig0, axs0 = plt.subplots(4,1, figsize=(6,8))
    fig1, axs1 = plt.subplots(4,1, figsize=(6,8))

    for axs, df in [(axs0, df0), (axs1, df1)]:
        ax0, ax1, ax2, ax3 = axs
        twin0 = ax0.twinx()
        twin1 = ax1.twinx()

        plot_sim(df, ["H", "E"], ax0)
        plot_sim(df, ["E/H"], twin0, "linear")
        
        plot_sim(df, ["N"],ax1,  "linear", ybottom=0)
        plot_sim(df, ["C"],twin1,"linear", ybottom=0)

        plot_sim(df, ["net $N$ uptake by $H$", "net $N$ uptake by $E$"], ax2, "linear")
        
        plot_sim(df, [r"$\mu_H$", r"$\rho_{food}$", r"$\rho_{photo}$"],  ax3, "linear")


        # plotting straight lines
        ax2.axhline(y=0, color = "k", dashes=(2,2))
        net_N_time = _find_crossing(df["net $N$ uptake by $H$"])

        for ax in (ax0, ax1, ax2, ax3):
            color = info.loc["H", "color"]
            ax.axvline(x=350, color=color, dashes = (1,1))
            ax.axvline(x=350+puls_duration, color=color, dashes = (1,1))

            for x0 in net_N_time:
                ax.axvline(x=x0, color="k",dashes=(1,1))

        ax1.text(x=350+puls_duration-15, y=0.01, s=r"$\longleftarrow " + f"{puls_var.name}=${puls_size}" )
        if net_N_time:
            ax2.text(x=net_N_time[0]+5, y =-0.004, s=r"$\leftarrow$ Switch in H's N-uptake")


        ctx = sns.plotting_context("talk")
        fsize = ctx["axes.labelsize"]

        ax1.legend(loc="upper right")
        twin1.legend(loc="right")

        ax0.set_ylabel("mol C/m$^2$", fontsize=fsize)
        twin0.set_ylabel("mol C/mol C", fontsize=fsize)
        ax1.set_ylabel("mol N/mol C", fontsize=fsize)
        twin1.set_ylabel("mol C/mol C", fontsize=fsize)
        ax1.set_xlabel("days", fontsize=fsize)
        ax2.set_ylabel("days$^{-1}$")
        ax3.set_ylabel("days$^{-1}$")
        ax3.set_xlabel("days")

    ## Rescaling axs1
    all_axs0 = axs0[0].figure.axes
    all_axs1 = axs1[0].figure.axes
    for i in range(len(all_axs0)):
        if i == 4: continue
        all_axs1[i].set_ylim(all_axs0[i].get_ylim())

    

    plt.show()

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
                ax.plot(bList[rowInd,0],bList[rowInd,2], color = "C2", marker = m, mfc = mfc, ms= ms, alpha=alpha, ls="-")
                ax.plot(bList[rowInd,0],bList[rowInd,1], color = "C0", marker = m, mfc = mfc, ms= ms, alpha=alpha, ls="-")
            else:
                ax.plot(bList[rowInd,0],bList[rowInd,2]/bList[rowInd,1], color = "C1", marker = m, mfc = mfc, ms= ms, alpha=alpha, ls="")

    #ax.legend()
    ax.set_xlabel(f"${para.name}$")
    ylabel = "$E^*/H^*$ (molar ratio)" if EtoH else r"$H^*,\,E^*$ mol C/m$^2$"
    ax.set_ylabel(ylabel)
    if save:
        saveName = para.name.replace("\\","")
        plt.savefig("figs/bifurs/bifur_" + saveName + ".png")



def runCollection(pList,save=False,preView=True,cons=[]):
    bifurList = { s: [1,1.4], to: [1,1.6], pmax: [0.01,1.0], uEmax: [0.005,0.07], uHmax: [0.0,0.01], KNE: [0.0001,0.12], KNH: [0.0,0.01], 
                 delC: [0.0,0.7], CI: [0.0,0.15], delN: [0.0,0.5], NI: [0.0,0.004], mH: [0.01,0.05], mE: [0.03,0.2], KCO2: [0.0001,0.06], rho0: [0,0.1],
                 QFood: [0,0.2], QE: [0.01,0.2], QH: [0.01,0.2], b: [0.01, 1], eps: [0,0.1] }
    fig, axs = plt.subplots(len(pList), len(pList[0]))

    for i in range(len(pList)):
        for j in range(len(pList[i])):
            para, span = pList[i][j], bifurList[pList[i][j]]
            b_list = makeSymbBifur(para, span, cons,ignore_H_lim=True)

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


def multBifur():
    bList = []
    uList = [0.03,0.05,0.06]
    for i in range(len(uList)):
        b =  makeSymbBifur(s,[0.9,1.5],[0,1],cons=[(uEmax,uList[i])])
        b[:,-2] = i
        if i == 0:
            bList = b
        else:
            bList = np.r_[bList, b]
    plotBifur(s,bList)
    plt.show()




#### Running stuff

#estab_sim()
#plot_phase()
#nitro_pulse_sim(1.0)
#starvation_sim()



big_sim(cons=[])






#runCollection([[s, to, pmax, rho0], [uEmax, KNE, NI, KCO2], [uHmax, KNH, CI, mH]], preView=True,save=True,
#              cons=[])


#runCollection([[NI,uHmax], [b,KNH]], save=False, cons=[     # [s, KNH, rho0], [uEmax, KNE, NI], [pmax, KCO2, CI]]
#
#         (pmax, 0.45), (uEmax,0.033), (uHmax,0.0045),
#
#         (KCO2, 0.02), (KNE, 0.03),   (KNH, 0.0001),
#
#         (NI, 0.001), (CI, 0.09)
#
#])

