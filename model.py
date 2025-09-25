## Model ODEs for endosymbiosis model
## Version 3.01, includes carbon pool and assumes fixed cell-quota for host and a dynamic DIN pool
import matplotlib.pyplot as plt
import numpy as np


def makeCons(changes=[]):
    dict = {"s": 1, "a":0.9, "b": 0.5, "pmax": 3, "KCO_2": 0.01, "d": 0.5, "CI": 0.1, "mE": 0.3, "mH": 0.03, 
            "KNE": 0.1, "uEmax" : 0.07, "KNH": 0.1, "uHmax" : 0.07, "QE": 0.15, "QH": 0.15, "emax": 0.5}

    for change in changes:
        dict[change[0]] = change[1]
    return dict


def makeFuncs(t, y, cD, envFlows):  
    H, E, N, C = y
    rhoDOC,rhoDON = envFlows

    a = cD["a"]
    b = cD["b"] # np.maximum(np.sin(2*np.pi*t),0)

    uH = cD["uHmax"] *(N)/( cD["KNH"] + N )
    uE = cD["uEmax"] *(N)/( cD["KNE"] + N )

    vE = cD["pmax"] * C/(cD["KCO_2"] + C)
    nE = (uE/vE)/cD["QE"]
    eE  = np.minimum(1/(cD["s"]+1), nE)

    rhoPhoto = (1-(1+cD["s"])*eE)*vE*E/H
    n = ( (rhoDON(t,y,cD) + uH)/(rhoDOC(t,y,cD)+rhoPhoto) )/cD["QH"]
    eH  = np.minimum(1/(cD["s"]+1), n)

    vH = b* ( cD["s"]*eH*(rhoDOC(t,y,cD) + rhoPhoto) + a*cD["mH"] + cD["s"]*eE*vE*E/H )
    rhoDIN = (1-eH/n)*(rhoDON(t,y,cD) + uH) + a*cD["mH"]*cD["QH"]

    muH = eH*(rhoDOC(t,y,cD) + rhoPhoto)
    muE = eE*vE



    return vE, vH, muH, muE, uH, uE, rhoPhoto, rhoDIN


def endo(t, y, cD, envFlows):
    H, E, N, C = y
    vE, vH, muH, muE, uH, uE, rhoPhoto, rhoDIN  = makeFuncs(t,y,cD,envFlows)

    dH = (muH-cD["mH"])*H
    dE = (muE-cD["mE"])*E

    dN = rhoDIN - uH - uE*E/H - 0.1*N
    dC  = vH - vE*E/H + cD["d"]*(cD["CI"]-C)

    return [dH,dE,dN,dC]


def _plotLimFac(t, y, cD, envFlows):
    H, E,  N, C = y
    t = sol.t
    rhoDOC, rhoDON = envFlows
    vE, vH, muH, muE, uH, uE, rhoPhoto, rhoDIN  = makeFuncs(t,y,cD,envFlows)

    fig, (ax1,ax2,ax3) = plt.subplots(nrows=3, ncols=1)

    ax1.plot(t,uE,"g--", label=r"$u_E$")
    ax1.plot(t,uE*E/H,"g", label=r"$u_E\frac{E}{H}$")
    ax1.plot(t,uH,"b", label="$u_H$")
    ax1.plot(t,rhoDIN,"b--", label=r"$\rho_{DIN}$")

    ax2.plot(t,muE,"g", label=r"$\mu_{E}$")
    ax2.plot(t,vE,"g--", label=r"$v_{E}$")

    ax3.plot(t,muH,"b", label=r"$\mu_{H}$")
    ax3.plot(t,rhoPhoto,"g--", label=r"$\rho_{photo}$")
    ax3.plot(t,rhoDOC(sol.t,sol.y,cD),"k--", label=r"$\rho_{Food}$")

    ax1.legend()
    ax2.legend()
    ax3.legend()

    ax3.set_xlabel("days")
    fig.supylabel(r"days$^{-1}$")



if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.integrate as integ
    import scipy.signal as signal

    def rhoDOC(t,y,cD):
        H, E, N, C = y
        return 0.03 *3* (1-H/166)
    def rhoDON(t,y,cD):
        H, E, N, C = y
        return rhoDOC(t,y,cD)*0.15
    
    y0 = [60, 0.01, 0.02, 0.1]
    tEnd = 800
    cD = makeCons([("s", 1.5), ("mH",0.03),("mE",0.10),("KNE",0.01), ("uEmax",0.06), ("KNH", 0.01), ("uHmax", 0.01),("pmax",0.5),("CI",0.2)])              
    sol = integ.solve_ivp(endo, y0=y0, t_span=[0,tEnd], args=(cD,[rhoDOC,rhoDON],), dense_output=False, vectorized=True, method="Radau", max_step = np.inf, rtol=1e-8, atol = 1e-8, events=[])
    print(sol)



    ### Plotting 
    H, E, QH, C = sol.y
    vE, vH, muH, muE, uH, uE, rhoPhoto, rhoDIN  = makeFuncs(sol.t,sol.y,cD,[rhoDOC,rhoDON])
    t = sol.t

    fig, (ax1,ax2,ax3) = plt.subplots(nrows=3, ncols=1)
    
    if y0[1]!=0:
        ax1.semilogy(t,E,"C2",label="E")
    ax1.semilogy(t,H,"C0",label="H")
    twin1 = ax1.twinx()
    twin1.plot(t,E/H,"gold", label = "$E/H$")
    
    ax2.plot(t,QH,"C0", label = "$N$")
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

    _plotLimFac(t, sol.y, cD, [rhoDOC,rhoDON])
    plt.show()