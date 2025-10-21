#Doing all the simulations, last updated 13

from analysisTools import *
import sympy as sp


###### Plotting bifurcation diagrams



###### Some simulations
def estabSim():
    y0 = [20, 0.001, 0.01, 0.14]   #Establishing under normal conditions
    tSpan = [0,500]
    t, y, funcs, cD = simSystem(y0,tSpan,cons=[("rho0",0.09)])
    plotSim(t,y,funcs)
    plt.savefig("figs/estab")
    plotCN(t,y,funcs,cD)
    plt.savefig("figs/estabCN")


    t2, y2, funcs2, cD2 = simSystem(y[:,-1],[500,800],cons=[("rho0",0.09),("s",1.5)])
    plotSim(t2,y2,funcs2)
    plt.savefig("figs/breakdown")
    plotCN(t2,y2,funcs2,cD2)
    plt.savefig("figs/breakdownCN")


def multEvents(y0,tSpan,cons=[],eventList=[]):
    t, y, funcs, cD = simSystem(y0,tSpan,cons)
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
    plotSim(t,y,funcs)
    plotCN(t,y,funcs,cD)
    return t,y,funcs



###### bifurcation diagram
def plotBifur(para,bList,ax=None, save=False):
    if ax == None:
        fig, ax = plt.subplots(1,1)

    mList = ["d", "^", "o", "s"] + ["x"]*10
    mfcList = ["none", None]

    for i in range(max(bList[:,-2])+1):
        for j in range(2):
            m, mfc = mList[i], mfcList[j]
            rowInd = (bList[:,-1]==j) & (bList[:,-2]==i)
            ax.plot(bList[rowInd,0],bList[rowInd,2], color = "C2", marker = m, mfc = mfc, ms= 5.0, alpha=0.6, ls="")
            ax.plot(bList[rowInd,0],bList[rowInd,1], color = "C0", marker = m, mfc = mfc, ms= 5.0, alpha=0.6, ls="")

    #ax.legend()
    ax.set_xlabel(f"${para.name}$")
    ax.set_ylabel("$H^*$, $E^*$ (mol C/m$^2$)")
    if save:
        saveName = para.name.replace("\\","")
        plt.savefig("figs/bifurs/num_bifur_" + saveName + ".png")



def runCollection(pList,save=False,preView=True,cons=[]):
    bifurList = { s: [0.9,1.5], to: [0.5,2], pmax: [0.01,1.5], uEmax: [0.015,0.09], uHmax: [0.0,0.006], KNE: [0.0,0.15], KNH: [0.0,0.7], 
                 delC: [0.0,0.7], CI: [0.0,0.15], delN: [0.0,0.7], NI: [0.0,0.02], mH: [0.01,0.06], KCO2: [0.0,0.05], rho0: [0,0.1]}
    fig, axs = plt.subplots(len(pList), len(pList[0]))

    for i in range(len(pList)):
        for j in range(len(pList[i])):
            para, span = pList[i][j], bifurList[pList[i][j]]
            bList = makeSymbBifur(para,span,cons)
            
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

#### Running stuff

#estabSim()

#multEvents([50, 0.01, 0.1, 0.02],[0,250],cons=[("s",1.0)],eventList=[
#   [[None, None, 0.5, None],200,[]]
#])
#plt.vlines(250, 0, 0.1,"grey")
#plt.text(250-2,4e-2,"N pulse ->",ha="right")



runCollection([[s, to, pmax], [CI,NI,rho0], [uEmax,uHmax,KNE]], preView=True,save=True)
