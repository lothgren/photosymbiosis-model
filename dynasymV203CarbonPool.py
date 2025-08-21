## Carbon pool version (felt cute)
## Version 3.00 ??
import matplotlib.pyplot as plt
import numpy as np


def makeCons(changes=[]):
    dict = {"s": 1, "pmax": 3, "KCO_2": 0.01, "b": 3, "d": 0.15, "KN": 0.05, "umax" : 0.03, "QHmin": 0.075, "QHmax":0.15, "QEmin": 0.03, "mE": 0.2, "mH": 0.03}

    for change in changes:
        dict[change[0]] = change[1]
    return dict


def makeFuncs(t, y, cD, envFlows):  
    L, H, QE, QH, C = y
    rhoDOC,rhoDON = envFlows

    eH  = cD["s"]*(1-cD["QHmin"]/QH) 
    eE  = (1-cD["QEmin"]/QE)

    mH = cD["mH"]
    mE = cD["mE"]

    vL = cD["pmax"] * C/(cD["KCO_2"]+C)
    rhoPhoto = (1-eE)*vL*L

    pH = eH*(rhoDOC(t,y,cD)+rhoPhoto)
    pL = eE*vL

    vH = cD["b"]*(1-2/cD["s"]*eH)*(rhoDOC(t,y,cD)+rhoPhoto) 
                                             
    uH = rhoDON(t,y,cD) * (1-((QH-cD["QHmin"])/(cD["QHmax"]-cD["QHmin"]))**2) 
    uE = cD["umax"] *(QH-cD["QHmin"])/( cD["KN"] + (QH-cD["QHmin"]) )           
    
    return pH, pL, mH, mE, uH, uE, vH, vL, rhoPhoto


def endo(t, y, cD, envFlows):
    L, H, QE, QH, C = y
    pH, pL, mH, mE, uH, uE, vH, vL, rhoPhoto  = makeFuncs(t,y,cD,envFlows)

    dL = pL*L - mE*L - (pH-mH)*L
    dH = pH*H - mH*H

    dQE = uE - pL*QE           
    dQH = uH - uE*L - pH*QH

    dC  = vH - vL*L - cD["d"]*C

    return [dL,dH,dQE,dQH,dC]


def plotLimFac(sol, cD, xSpan, envFlows):
    L, H, QE, QH, C = sol.y
    t = sol.t
    pH, pL, mH, mE, uH, uE, vH, vL, rhoPhoto  = makeFuncs(sol.t,sol.y,cD,envFlows)
    fig, (ax1,ax2,ax3) = plt.subplots(nrows=3, ncols=1)

    ax1.plot(sol.t,uE,"g--", label=r"$u_E$")
    ax1.plot(sol.t,uE*sol.y[0],"g", label=r"$u_E\frac{E}{H}$")
    ax1.plot(sol.t,uH,"b--", label="$u_H$")
    ax1.plot(sol.t,pH*sol.y[3],"b", label=r"$\mu_HQ_H$")

    ax2.plot(sol.t, vH,"b", label=r"$v_{H}$")
    ax2.plot(sol.t, vL*L,"g--", label=r"$v_{E}L$")
    ax2.plot(sol.t, pL,"g", label=r"$\mu_{E}$")  

    ax3.plot(sol.t,pH,"b", label=r"$\mu_{H}$")
    ax3.plot(sol.t,rhoPhoto,"g--", label=r"$\rho_{photo}$")
    ax3.plot(sol.t,rhoDOC(sol.t,sol.y,cD),"k--", label=r"$Food$")

    

    ax1.set_title("") 
    ax1.set_ylabel(r"days$^{-1}$")
    ax2.set_xlabel("days")
    ax1.set_xlim(xSpan)
    ax2.set_xlim(xSpan)
    ax3.set_xlim(xSpan)
    ax1.legend()
    ax2.legend()
    ax3.legend()
  

def extinctE(t,y,cD,envFlows):
    return y[0]-1e-10
extinctE.terminal = True


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.integrate as integ
    import scipy.signal as signal

    def rhoDOC(t,y,cD):
        E, H, QE, QH, C = y
        return 0.03 *2* (1-H/166)
    def rhoDON(t,y,cD):
        E, H, QE, QH, C = y
        return rhoDOC(t,y,cD)*0.15 + cD["mE"]*0.0
    
    y0 = [1e-4, 60, 0.04, 0.12, 1e-1]
    tEnd = 1000
    cD = makeCons([("s",0.2)])
    sol = integ.solve_ivp(endo, y0=y0, t_span=[0,tEnd], args=(cD,[rhoDOC,rhoDON],), dense_output=False,
                           method="Radau", max_step = np.inf, rtol=1e-8, atol = 1e-8, events=[extinctE])
    print(sol)


    fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1)
    if sol.t[-1]==tEnd:
        xSpan = [-5,tEnd]
    else:
        xSpan = [0,sol.t[-1]]
    

    ax1.semilogy(sol.t,sol.y[0]*sol.y[1],"C2",label="E")
    ax1.semilogy(sol.t,sol.y[1],"C0",label="H")
    

    #hTest = y0[1]*np.exp(-cD["mH"]*sol.t)    
    #ax1.semilogy(sol.t,hTest,"b--",label="test H")
    twin2 = ax2.twinx()
    twin2.plot(sol.t,sol.y[4],"k--",label="C")
    ax2.plot(sol.t,sol.y[2],"C2", label = "$Q_E$")
    ax2.plot(sol.t,sol.y[3],"C0", label = "$Q_H$")
    ax2.plot(sol.t,sol.y[0],"gold", label = "$E/H$")
 
    ax1.set_ylabel(r"mol C /m$^2$")
    ax2.set_ylabel("molar ratio")
    ax2.set_xlabel("d")
    ax1.set_xlim(xSpan)
    ax2.set_xlim(xSpan)
    if sol.t_events[0].size>0:
        ax2.set_ylim([None,0.5])
    ax1.legend()
    ax2.legend()

    plotLimFac(sol,cD,xSpan,[rhoDOC,rhoDON])
    plt.show()