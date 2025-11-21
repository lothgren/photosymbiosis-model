#Doing all the simulations, last updated 13

from analysisTools import *
import sympy as sp


###### Plotting bifurcation diagrams



###### Some simulations
def multEvents(y0,tSpan,cons=[],eventList=[]):
    t, y, funcs, cD = simSystem(y0,tSpan,cons)
    y[1,y[1,:]==0] = np.nan
    y0Old, spanOld = y[:,-1], tSpan
    
    for event in eventList:
        ny0, newSpan, newCons = event
        
        for i in range(len(y0)):
            if ny0[i] == None:
                ny0[i] = y0Old[i]
        
        if not isinstance(newSpan,list):
            newSpan = [spanOld[1],spanOld[1]+newSpan]

        t2, y2, funcs2, cD2 = simSystem(ny0, newSpan, newCons)
        t = np.append(t,t2)
        y = np.c_[y, y2]
        funcs = np.c_[funcs, funcs2]
        y0Old, spanOld = y2[:,-1], newSpan
    return t,y,funcs,cD


def estabSim():
    _, y_start, _, _ = simSystem([55, 0, 0.05, 0.16], [0,100])
    y0 = y_start[:,-1]   #Establishing under normal conditions
    y0[1] = 1e-1
    tSpan = [0,500]
    t, y, funcs, cD = multEvents(y0,tSpan,cons=[])
    axs = plot_sim_2(t,y,funcs,cD)
    

def nitro_pulse_sim():
    s_value = 1.20
    _, y_start, _, _ = simSystem([136.137566137566, 22.6164549610313, 0.0171428571428571, 0.0294087258680283], [0,500],cons=[(s,s_value)])
    y0 = y_start[:,-1]   #Establishing under normal conditions
    tSpan = [0,50]
    t, y, funcs, cD = multEvents(y0,tSpan,cons=[(s,s_value)], eventList=[
        [[None, None, 0.04, None],200,[(s,s_value)]]
    ])
    axs = plot_sim_2(t,y,funcs,cD)#


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
    

###### bifurcation diagram
def plotBifur(para,bList,ax=None, save=False, EtoH = False):
    if ax == None:
        fig, ax = plt.subplots(1,1)

    mList = ["^", "d", "o", "s"] + ["x"]*10
    mfcList = ["none", None]

    ms, alpha = 5.0, 0.7

    for i in range(max(bList[:,-2])+1):
        for j in range(2):
            m, mfc = mList[i], mfcList[j]
            rowInd = (bList[:,-1]==j) & (bList[:,-2]==i)
            if not EtoH:
                ax.plot(bList[rowInd,0],bList[rowInd,2], color = "C2", marker = m, mfc = mfc, ms= ms, alpha=alpha, ls="")
                ax.plot(bList[rowInd,0],bList[rowInd,1], color = "C0", marker = m, mfc = mfc, ms= ms, alpha=alpha, ls="")
            else:
                ax.plot(bList[rowInd,0],bList[rowInd,2]/bList[rowInd,1], color = "C1", marker = m, mfc = mfc, ms= ms, alpha=alpha, ls="")

    #ax.legend()
    ax.set_xlabel(f"${para.name}$")
    ylabel = "$E^*/H^*$ (molar ratio)" if EtoH else r"$H^*,\,E^*$ mol C/m$^2$"
    ax.set_ylabel(ylabel)
    if save:
        saveName = para.name.replace("\\","")
        plt.savefig("figs/bifurs/num_bifur_" + saveName + ".png")



def runCollection(pList,save=False,preView=True,cons=[]):
    bifurList = { s: [1,2], to: [0.5,2], pmax: [0.01,1.25], uEmax: [0.001,0.07], uHmax: [0.0,0.04], KNE: [0.0001,0.2], KNH: [0.0,0.005], 
                 delC: [0.0,0.7], CI: [0.0,0.15], delN: [0.0,0.5], NI: [0.0,2e-3], mH: [0.01,0.06], mE: [0.03,0.2], KCO2: [0.0,0.1], rho0: [0,0.1],
                 QFood: [0,0.2], QE: [0.01,0.2], QH: [00.01,0.2] }
    fig, axs = plt.subplots(len(pList), len(pList[0]))

    for i in range(len(pList)):
        for j in range(len(pList[i])):
            para, span = pList[i][j], bifurList[pList[i][j]]    
            b1, b2 = makeSymbBifur(para,span,[0,1],cons), makeSymbBifur(para,span,[0,0],cons)
            b3, b4 = [],[]#makeSymbBifur(para,span,[1,0],cons), makeSymbBifur(para,span,[1,1],cons)
            bList = b1
            for b in [b2,b3,b4]:
                if not np.ndim(b) <= 1:
                    bList = np.r_[bList, b]

            if preView:
                if np.ndim(np.squeeze(pList)) >= 2:
                    ax = axs[i,j] 
                elif np.ndim(np.squeeze(pList)) == 1: 
                    ax = axs[i+j]
                else:
                    ax = axs
                plotBifur(para,bList,ax=ax,save=False)
            
            if save:
                plotBifur(para,bList,ax=None,save=True)
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

#estabSim()
#plot_phase()
#nitro_pulse_sim()
#starvation_sim()
#plt.show()


#runCollection([[s, to, pmax, rho0], [uEmax, KNE, NI, KCO2], [uHmax, KNH, CI, mH]], preView=True,save=True,
#              cons=[])


runCollection([[s, to, pmax, rho0], [uEmax, KNE, NI, KCO2], [uHmax, KNH, CI, mH]], save=True, cons=[
        (b,0.5), (s,1),
        (pmax,0.45), (KCO2,0.01), (uEmax,0.02 ), (KNE,0.01), (mE,0.03), 
                                  (uHmax,0.005), (KNH,0.0001), (mH,0.03),
        (NI,0.001), (delN,0.15), (CI,0.15), (delC,0.4), (rho0,0.07)
])

