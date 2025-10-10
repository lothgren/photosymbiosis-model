## Model ODEs for endosymbiosis model
## Version 3.01, includes carbon pool and assumes fixed cell-quota for host and a dynamic DIN pool
import matplotlib.pyplot as plt
import numpy as np


def makeCons(changes=[]):
    dict = {"s": 1, "to": 1, "b": 0.5, "pmax": 0.5, "KCO_2": 0.01, "dC": 0.5, "CI": 0.1, "mE": 0.05, "mH": 0.03, "rho0": 0.03*3, "HCap": 166,
            "dN": 0.2, "NI": 0.005, "KNE": 0.05, "uEmax" : 0.04, "KNH": 0.01, "uHmax" : 0.04, "QE": 0.15, "QH": 0.15, "QFood": 0.15}

    for change in changes:
        dict[change[0]] = change[1]
    return dict


def makeFuncs(t, y, cD):  
    H, E, N, C = y
    rhoDOC = np.maximum(0,cD["rho0"]*(1-H/cD["HCap"]))
    rhoDON = rhoDOC*cD["QFood"]

    b = cD["b"]

    mH = cD["mH"]*cD["to"] + H*0
    mE = cD["mE"]*cD["to"] + H*0

    uH = cD["uHmax"] *(N)/( cD["KNH"] + N )
    uE = cD["uEmax"] *(N)/( cD["KNE"] + N )

    pE = cD["pmax"] * C/(cD["KCO_2"] + C)
    nE = (uE/pE)/cD["QE"]
    eE  = np.minimum(1/(cD["s"]+1), nE)

    rhoPhoto = (1-(1+cD["s"])*eE)*pE*E/H
    nH = ( (rhoDON + uH)/(rhoDOC+rhoPhoto) )/cD["QH"]
    eH  = np.minimum(1/(cD["s"]+1), nH)

    rH = b* ( cD["s"]*eH*(rhoDOC + rhoPhoto) + mH)
    rE = b* ( cD["s"]*eE*pE + mE )

    rhoDIN = (1-eH/nH)*(rhoDON + uH) + mH*cD["QH"] + mE*cD["QE"]*E/H

    muH = eH*(rhoDOC + rhoPhoto)
    muE = eE*pE


    return rE, rH, pE, muH, muE, uH, uE, rhoPhoto, rhoDIN, mH, mE, rhoDOC, rhoDON


def endo(t, y, cD):
    H, E, N, C = y
    rE, rH, pE, muH, muE, uH, uE, rhoPhoto, rhoDIN, mH, mE, rhoDOC, rhoDON  = makeFuncs(t,y,cD)

    dH = (muH-mH)*H
    dE = (muE-mE)*E

    dN = rhoDIN - uH - uE*E/H + cD["dN"]*(cD["NI"]-N)
    dC  = rH + rE*E/H - pE*E/H + cD["dC"]*(cD["CI"]-C)

    return [dH,dE,dN,dC]


def symbDeath(t,y,cD):
    return y[1]-1e-10
symbDeath.terminal = True

def loadStop(t,y,cD):
    return 2-y[1]/y[0]
loadStop.terminal = True


def _plotLimFac(t, y, cD):
    H, E,  N, C = y
    rE, rH, pE, muH, muE, uH, uE, rhoPhoto, rhoDIN, mH, mE, rhoDOC, rhoDON  = makeFuncs(t,y,cD)

    fig, (ax1,ax2,ax3) = plt.subplots(nrows=3, ncols=1)

    ax1.plot(t,N,"C0", label = "$N$")
    twin1 = ax1.twinx()
    twin1.plot(t,C,"k--",label="C")

    ax2.plot(t,uE,"g--", label=r"$u_E$")
    ax2.plot(t,uE*E/H,"g", label=r"$u_E\frac{E}{H}$")
    ax2.plot(t,rhoDIN-uH,"b", label=r"$\rho_{DIN}-u_H$")
    ax2.plot(t,cD["dN"]*(cD["NI"]-N), "r--", label=r"$\delta_N (N_I-N)$")
 
    ax3.plot(t, rH,"b", label=r"$r_{H}$")
    ax3.plot(t, rE*E/H,"b--", label=r"$r_{E}\frac{E}{H}$")
    ax3.plot(t, pE*E/H,"g--", label=r"$p_{E}\frac{E}{H}$")
    ax3.plot(t, cD["dC"]*(cD["CI"]-C),"k--", label=r"$\delta_C (C_I-C)$")

    ax1.legend(loc="upper right")
    twin1.legend(loc="lower right")
    ax2.legend()
    ax3.legend()

    ax1.set_ylabel("mol N/mol C")
    twin1.set_ylabel("mol CO$_2$/mol C")
    ax2.set_ylabel("mol N/mol C/d")
    ax3.set_ylabel("mol C/mol C/d")
    ax3.set_xlabel("days")
    



if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.integrate as integ
    import scipy.signal as signal
    
    y0 = [17,0.0001,0.05,0.16]  #[1.56068376e+02, 2.58806457e+01, 1.15384615e-02, 6.07692308e-02]
    tEnd = 500
    cD = makeCons([("pmax",0.5),("KNE",0.004)])
    sol = integ.solve_ivp(endo, y0=y0, t_span=[0,tEnd], args=(cD,), dense_output=False, vectorized=True, method="Radau", max_step = np.inf, rtol=1e-8, atol = 1e-8, events=[])
    print(sol)
    print(sol.y[:,-1])

    ### Plotting 
    H, E, N, C = sol.y
    rE, rH, pE, muH, muE, uH, uE, rhoPhoto, rhoDIN, mH, mE, rhoDOC, rhoDON = makeFuncs(sol.t,sol.y,cD)
    t = sol.t

    fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1)
    
    if y0[1]!=0:
        ax1.semilogy(t,E,"C2",label="E")
    ax1.semilogy(t,H,"C0",label="H")
    twin1 = ax1.twinx()
    twin1.plot(t,E/H,"gold", label = "$E/H$")
    
    twin2 = ax2.twinx()
    twin2.plot(t,muE,"g", label=r"$\mu_{E}$")
    twin2.plot(t,pE,"g--", label=r"$p_{E}$")
    ax2.plot(t,muH,"b", label=r"$\mu_{H}$")
    ax2.plot(t,rhoPhoto,"r--", label=r"$\rho_{photo}$")
    ax2.plot(t,rhoDOC,"k--", label=r"$\rho_{Food}$")


    ax1.set_ylabel(r"mol C /m$^2$")
    twin1.set_ylabel("E biomass/H iomass")
    ax2.set_ylabel(r"d$^{-1}$")
    twin2.set_ylabel(r"d$^{-1}$")
    ax2.set_xlabel("d")


    ax1.legend()
    twin1.legend()
    ax2.legend(loc="upper right")
    twin2.legend(loc="lower right")


    _plotLimFac(t, sol.y, cD)
    plt.show()