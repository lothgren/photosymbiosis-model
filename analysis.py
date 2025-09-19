#Doing all the simulations, last updated 13

from analysisTools import *
import sympy as sp

##### Functions and constants
## Functions describing environmental flow of carbon and nutrients

def rhoDOC(t,y,cD):
        H, E, QE, QH, C = y
        return 0.03 *3* (1-H/166)
def rhoDON(t,y,cD):
    H, E, QE, QH, C = y
    return rhoDOC(t,y,cD)*0.15 + cD["mH"]*QH*0.9


###### Plotting bifurcation diagrams



###### Some simulations

def stepWiseSim():
    y0 = [5, 0, 0.04, 0.12, 0.0]                              # Plot establishing of host
    cD = makeCons([("s", 1), ("mH",0.03),("mE",0.3),("KN",0.05),("umax",0.03),("pmax",3),("CI",0.2)]) 
    sol1 = integ.solve_ivp(endo, y0=y0, t_span=[0,500], args=(cD,[rhoDOC,rhoDON],), dense_output=False, method="Radau", max_step = np.inf, rtol=1e-8, atol = 1e-8, events=[])
    sol1.y[1,:] = np.array([None]*len(sol1.y[1,:]))
    sol1.y[2,:] = np.array([None]*len(sol1.y[1,:]))
    
    ny0 = [sol1.y[0,-1], 1e-2, 0.04, sol1.y[3,-1], sol1.y[4,-1]]
    sol2 = integ.solve_ivp(endo, y0=ny0, t_span=[500,800], args=(cD,[rhoDOC,rhoDON],), dense_output=False, method="Radau", max_step = np.inf, rtol=1e-8, atol = 1e-8, events=[])

    cD = makeCons([("s", 2), ("mH",0.03),("mE",0.3),("KN",0.05),("umax",0.03),("pmax",3),("CI",0.2)])
    ny0 = sol2.y[:,-1]
    sol3 = integ.solve_ivp(endo, y0=ny0, t_span=[800,1200], args=(cD,[rhoDOC,rhoDON],), dense_output=False, method="Radau", max_step = np.inf, rtol=1e-8, atol = 1e-8, events=[])



    ### Plotting 
    Y = np.c_[sol1.y, sol2.y, sol3.y]
    H, E, QE, QH, C = Y
    t = np.append(sol1.t,np.append(sol2.t,sol3.t))


    fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1)
    
    ax1.axvline(500, color="k",dashes=(0.7,0.7),alpha=0.8)
    ax1.axvline(800, color="k",dashes=(0.7,0.7),alpha=0.8)
    ax1.axvline(598, color="k",dashes=(0.7,0.7),alpha=0.8)
    ax2.axvline(598, color="k",dashes=(0.7,0.7),alpha=0.8)
    ax2.axvline(500, color="k",dashes=(0.7,0.7),alpha=0.8)
    ax2.axvline(800, color="k",dashes=(0.7,0.7),alpha=0.8)
    ax1.text(500-2,1e-2,"E infection ->",ha="right")
    ax1.text(800+2,1e-0,"<- Increase C demand",ha="left")
    
    ax1.semilogy(t,E,"C2",label="E")
    ax1.semilogy(t,H,"C0",label="H")
    
    twin2 = ax2.twinx()
    twin2.plot(t,C,"k",label="C")
    ax2.plot(t,QE,"C2", label = "$Q_E$")
    ax2.plot(t,QH,"C0", label = "$Q_H$")
    ax2.plot(t,E/H,"gold", label = "$E/H$")
 
    ax1.set_ylabel(r"mol C /m$^2$")
    ax2.set_ylabel("molar ratio")
    twin2.set_ylabel("$CO_2$ per host biomass")
    ax2.set_xlabel("days")
   
    ax1.legend()
    ax2.legend(loc="upper right")
    twin2.legend(loc="lower right")
    plt.show()

stepWiseSim()    

