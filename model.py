## Model ODEs for endosymbiosis model
## Version 3.00, includes carbon pool
import matplotlib.pyplot as plt
import numpy as np


def makeCons(changes=[]):
    dict = {"s": 1, "pmax": 1, "KCO_2": 0.01, "b": 0.5, "d": 0.5, "CI": 0.5, "KN": 0.1, "umax" : 0.07, "QHmin": 0.075, "QHmax":0.15, "QEmin": 0.03, "mE": 0.3, "mH": 0.03}

    for change in changes:
        dict[change[0]] = change[1]
    return dict


def makeFuncs(t, y, cD, envFlows):  
    H, E, QE, QH, C = y
    rhoDOC,rhoDON = envFlows

    eH  = cD["s"]*(1-cD["QHmin"]/QH)
    eE  = (1-cD["QEmin"]/QE)

    vE = cD["pmax"] * C/(cD["KCO_2"] + C)
    rhoPhoto = (1-eE)*vE*E/H
    vH = cD["b"]*(1-eH)*(rhoDOC(t,y,cD) + rhoPhoto) + 0.9*cD["mH"]

    muH = eH*(rhoDOC(t,y,cD) + rhoPhoto)
    muE = eE*vE

    uH = rhoDON(t,y,cD) * (1-((QH-cD["QHmin"])/(cD["QHmax"]-cD["QHmin"]))**10)
    uE = cD["umax"] *(QH-cD["QHmin"])/( cD["KN"] + (QH-cD["QHmin"]) )

    return vE, vH, muH, muE, uH, uE, rhoPhoto


def endo(t, y, cD, envFlows):
    H, E, QE, QH, C = y
    vE, vH, muH, muE, uH, uE, rhoPhoto  = makeFuncs(t,y,cD,envFlows)

    dH = (muH-cD["mH"])*H
    dE = (muE-cD["mE"])*E

    dQE = uE - muE*QE
    dQH = uH - uE*E/H - muH*QH
    dC  = vH - vE*E/H + cD["d"]*(cD["CI"]-C)

    return [dH,dE,dQE,dQH,dC]


def _plotLimFac(t, y, cD, envFlows):
    H, E, QE, QH, C = y
    t = sol.t
    rhoDOC,rhoDON = envFlows
    vE, vH, muH, muE, uH, uE, rhoPhoto  = makeFuncs(t,y,cD,envFlows)

    fig, (ax1,ax2,ax3) = plt.subplots(nrows=3, ncols=1)

    ax1.plot(t,uE,"g--", label=r"$u_E$")
    ax1.plot(t,uE*E/H,"g", label=r"$u_E\frac{E}{H}$")
    ax1.plot(t,uH,"b--", label="$u_H$")
    ax1.plot(t,muH*QH,"b", label=r"$\mu_HQ_H$")

    ax2.plot(t, vH,"b", label=r"$v_{H}$")
    ax2.plot(t, vE*E/H,"g--", label=r"$v_{E}\frac{E}{H}$")
    ax2.plot(t, cD["d"]*(cD["CI"]-C),"k--", label=r"$\delta (C_I-C)$")

    ax3.plot(t,muH,"b", label=r"$\mu_{H}$")
    ax3.plot(t,rhoPhoto,"g--", label=r"$\rho_{photo}$")
    ax3.plot(t,rhoDOC(sol.t,sol.y,cD),"k--", label=r"$Food$")

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
        H, E, QE, QH, C = y
        return 0.03 *4* (1-H/166)
    def rhoDON(t,y,cD):
        H, E, QE, QH, C = y
        return rhoDOC(t,y,cD)*0.15 + cD["mH"]*QH*0.9
    
    y0 = [60, 0.1, 0.04, 0.12, 0.1]
    tEnd = 500
    cD = makeCons([("s",1), ("mH",0.03),("mE",0.3),("KN",0.02),("umax",0.03),("pmax",3),("CI",0.5)])                  #Read up on vectorized!
    sol = integ.solve_ivp(endo, y0=y0, t_span=[0,tEnd], args=(cD,[rhoDOC,rhoDON],), dense_output=False, vectorized=True, method="Radau", max_step = np.inf, rtol=1e-8, atol = 1e-8, events=[])
    print(sol)



    ### Plotting 
    H, E, QE, QH, C = sol.y
    t = sol.t

    fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1)
    
    if y0[1]!=0:
        ax1.semilogy(t,E,"C2",label="E")
    ax1.semilogy(t,H,"C0",label="H")
    
    twin2 = ax2.twinx()
    twin2.plot(t,C,"k--",label="C")
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

    _plotLimFac(t, sol.y, cD, [rhoDOC,rhoDON])
    plt.show()