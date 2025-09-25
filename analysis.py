#Doing all the simulations, last updated 13

from analysisTools import *
import sympy as sp

##### Functions and constants
## Functions describing environmental flow of carbon and nutrients

def rhoDOC(t,y,cD):
    H, E, N, C = y
    return 0.03 *3* (1-H/166)
def rhoDON(t,y,cD):
    H, E, N, C = y
    return rhoDOC(t,y,cD)*0.2


###### Plotting bifurcation diagrams



###### Some simulations

def stepWiseSim():
    y0 = [5, 0, 0.001, 0.001]                              # Plot establishing of host
    cD = makeCons([("s", 1), ("mH",0.03),("mE",0.10),("KNE",0.05), ("uEmax",0.06), ("KNH", 0.01), ("uHmax", 0.01),("pmax",0.5),("CI",0.2)]) 
    sol1 = integ.solve_ivp(endo, y0=y0, t_span=[0,500], args=(cD,[rhoDOC,rhoDON],), dense_output=False, method="Radau", max_step = np.inf, rtol=1e-8, atol = 1e-8, events=[])
    sol1.y[1,:] = np.array([np.nan]*len(sol1.y[1,:]))
    
    ny0 = [sol1.y[0,-1], 1e-2, sol1.y[2,-1], sol1.y[3,-1]]
    sol2 = integ.solve_ivp(endo, y0=ny0, t_span=[500,1000], args=(cD,[rhoDOC,rhoDON],), dense_output=False, method="Radau", max_step = np.inf, rtol=1e-8, atol = 1e-8, events=[])

    cD["s"] = 1.7
    ny0 = sol2.y[:,-1]
    sol3 = integ.solve_ivp(endo, y0=ny0, t_span=[1000,1500], args=(cD,[rhoDOC,rhoDON],), dense_output=False, method="Radau", max_step = np.inf, rtol=1e-8, atol = 1e-8, events=[])



    ### Plotting 
    Y = np.c_[sol1.y, sol2.y, sol3.y]
    H, E, N, C = Y
    t = np.append(sol1.t,np.append(sol2.t,sol3.t))
    vE, vH, muH, muE, uH, uE, rhoPhoto, rhoDIN  = makeFuncs(t,Y,cD,[rhoDOC,rhoDON])


    fig, (ax1,ax2,ax3) = plt.subplots(nrows=3, ncols=1)
    
    ax1.axvline(500, color="k",dashes=(0.7,0.7),alpha=0.8)
    ax1.axvline(1000, color="k",dashes=(0.7,0.7),alpha=0.8)
    ax2.axvline(500, color="k",dashes=(0.7,0.7),alpha=0.8)
    ax2.axvline(1000, color="k",dashes=(0.7,0.7),alpha=0.8)
    ax3.axvline(500, color="k",dashes=(0.7,0.7),alpha=0.8)
    ax3.axvline(1000, color="k",dashes=(0.7,0.7),alpha=0.8)
    ax1.text(500-2,1e-2,"E infection ->",ha="right")
    ax1.text(1000+2,1e-0,"<- Increase C demand",ha="left")
    
    
    ax1.semilogy(t,E,"C2",label="E")
    ax1.semilogy(t,H,"C0",label="H")
    twin1 = ax1.twinx()
    twin1.plot(t,E/H,"gold", label = "$E/H$")
    
    ax2.plot(t,N,"C0", label = "$N$")
    twin2 = ax2.twinx()
    twin2.plot(t,C,"k--",label="C")
 
    ax3.plot(t, vH,"b", label=r"$v_{H}$")
    ax3.plot(t, vE*E/H,"g--", label=r"$v_{E}\frac{E}{H}$")
    ax3.plot(t, cD["d"]*(cD["CI"]-C),"k--", label=r"$\delta (C_I-C)$")

    ax1.set_ylabel(r"mol C /m$^2$")
    twin1.set_ylabel("E biomass/H iomass")
    ax2.set_ylabel("mol N/mol C")
    twin2.set_ylabel("mol $CO_2$/H biomass")
    ax3.set_ylabel(r"C uptake (d$^{-1}$)")
    ax3.set_xlabel("d")


    ax1.legend()
    twin1.legend()
    ax2.legend(loc="upper right")
    twin2.legend(loc="lower right")
    ax3.legend()
    plt.show()


def bifurPlotting():
    fig, axs = plt.subplots(2,3)
    
    cons = [("s", 1), ("mH",0.03),("mE",0.13),("KNE",0.05), ("uEmax",0.06), ("KNH", 0.01), ("uHmax", 0.01),("pmax",0.35),("CI",0.2)]
    
    plotBifur("pmax", [0.0,0.8], [rhoDOC,rhoDON], cons, bFunc=bifur)





#### Running stuff

stepWiseSim()    

