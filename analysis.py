#Doing all the simulations, last updated 13

from analysisTools import *
import sympy as sp

##### Functions and constants


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



def stepWiseSim():
    cons = []
    y0 = [5, 0, 0.001, 0.001]                 # Plot establishing of host
    t1, y1, funcs1, cD = simSystem(y0, [0,500],cons=cons)
    
    ny0 = y1[:,-1]
    ny0[1] = 1e-2                             # Infection and establishing of endosymbiont
    t2, y2, funcs2, cD = simSystem(ny0, [500,1000],cons=cons)

    cons += [("s",1.5)]
    ny0 = y2[:,-1]                            # Increased energy demand
    t3, y3, funcs3, cD = simSystem(ny0, [1000,1500],cons=cons)

    ny0 = y3[:,-1]
    ny0[2] = 0.05                             # N pulse
    t4, y4, funcs4, cD = simSystem(ny0, [1500,2000],cons=cons)
    

    ### Plotting 
    Y = np.c_[y1, y2, y3, y4]
    H, E, N, C = Y
    t = np.append(t1,np.append(t2,np.append(t3,t4)))
    rE, rH, pE, muH, muE, uH, uE, rhoPhoto, rhoDIN, mH, mE, rhoDOC, rhoDON = np.c_[funcs1, funcs2, funcs3, funcs4]


    fig, axs = plt.subplots(nrows=2, ncols=1)
    
    axs[0].axvline(500, color="k",dashes=(0.7,0.7),alpha=0.8)
    axs[1].axvline(500, color="k",dashes=(0.7,0.7),alpha=0.8)
    axs[0].text(500-2,5e-2,"E infection ->",ha="right")

    axs[0].axvline(1000, color="k",dashes=(0.7,0.7),alpha=0.8)
    axs[1].axvline(1000, color="k",dashes=(0.7,0.7),alpha=0.8)
    axs[0].text(1000+2,1e-0,"<- Increase C demand",ha="left")

    axs[0].axvline(1500, color="k",dashes=(0.7,0.7),alpha=0.8)
    axs[1].axvline(1500, color="k",dashes=(0.7,0.7),alpha=0.8)
    axs[0].text(1500+2,5e-2,"<- N pulse",ha="left")
    
    
    axs[0].semilogy(t,E,"C2",label="E")
    axs[0].semilogy(t,H,"C0",label="H")
    twin0 = axs[0].twinx()
    twin0.plot(t,E/H,"gold", label = "$E/H$")
    #twin0.set_ylim(None,1.1)
    
    
    twin1 = axs[1].twinx()
    twin1.plot(t,muE,"g", label=r"$\mu_{E}$")
    twin1.plot(t,pE,"g--", label=r"$p_{E}$")
    axs[1].plot(t,muH,"b", label=r"$\mu_{H}$")
    axs[1].plot(t,rhoPhoto,"r--", label=r"$\rho_{photo}$")
    axs[1].plot(t,rhoDOC,"k--", label=r"$\rho_{Food}$")


    axs[0].set_ylabel(r"mol C /m$^2$")
    twin0.set_ylabel("E biomass/H iomass")
    axs[1].set_ylabel(r"d$^{-1}$")
    twin1.set_ylabel(r"d$^{-1}$")
    axs[1].set_xlabel("d")

    axs[0].legend()
    twin0.legend()
    axs[1].legend(loc="upper right")
    twin1.legend(loc="lower right")

    plt.savefig("figs/estab2Breakdown.png")


    ######################################### second plot ######################################
    fig2, axs2 = plt.subplots(nrows=3, ncols=1)

    axs2[0].plot(t,N,"C0", label = "$N$")
    twin20 = axs2[0].twinx()
    twin20.plot(t,C,"k--",label="C")

    axs2[1].plot(t,uE,"g--", label=r"$u_E$")
    axs2[1].plot(t,uE*E/H,"g", label=r"$u_E\frac{E}{H}$")
    axs2[1].plot(t,rhoDIN-uH,"b--", label=r"$\rho_{DIN}$")
    axs2[1].plot(t, cD["dN"]*(cD["NI"]-N),"r--", label=r"$\delta_N (N_I-N)$")
 
    axs2[2].plot(t, rH,"b", label=r"$r_{H}$")
    axs2[2].plot(t, rE*E/H,"b--", label=r"$r_{E}\frac{E}{H}$")
    axs2[2].plot(t, pE*E/H,"g--", label=r"$p_{E}\frac{E}{H}$")
    axs2[2].plot(t, cD["dC"]*(cD["CI"]-C),"k--", label=r"$\delta_C (C_I-C)$")

    axs2[0].legend()
    axs2[1].legend()
    twin0.legend()
    axs2[2].legend()

    axs2[0].set_ylabel("mol N/mol C")
    twin20.set_ylabel("mol CO$_2$/mol C")
    axs2[1].set_ylabel("mol N/mol C/d")
    axs2[2].set_ylabel("mol C/mol C/d")
    axs2[2].set_xlabel("days")

    plt.savefig("figs/estab2BreakdownCN.png")


def bifurPlotting():
    fig, axs = plt.subplots(3,3)
    
    cons = []
    
    
    plotBifur("s", [0.5,2.5], cons, bFunc=bifur, save=True, ax=axs[0,0])
    plotBifur("to", [0.5,2], cons, initVal=None,bFunc=bifur, save=True, ax=axs[1,0])
    plotBifur("pmax", [2,0], cons, bFunc=bifur, save=True, ax=axs[2,0])

    plotBifur("uEmax", [0.01,0.07], cons, bFunc=bifur, save=True, ax=axs[0,1])
    plotBifur("KNE", [0.3,0.001], cons, bFunc=bifur, save=True, ax=axs[1,1])
    plotBifur("dN", [0.35,0.001], cons, bFunc=bifur, save=True, ax=axs[2,1])

    plotBifur("uHmax", [0.07,0], cons, bFunc=bifur, save=True, ax=axs[0,2])
    plotBifur("KNH", [0.3,0.001], cons, bFunc=bifur, save=True, ax=axs[1,2])    
    plotBifur("dC", [0.6,0], cons, bFunc=bifur, save=True, ax=axs[2,2])


    
    




#### Running stuff

#estabSim()

multEvents([50, 0.01, 0.1, 0.02],[0,400],cons=[("s",1.0)],eventList=[
   [[None, None, None, None],200,[("NI",0.075)]]
])
plt.vlines(400, 0, 0.1,"grey")
plt.text(400-2,5e-2,"NI increase ->",ha="right")


#stepWiseSim()
#bifurPlotting()

plt.show()
