## Carbon pool version (felt cute)
## Version 3.00 ??
import matplotlib.pyplot as plt
import numpy as np


def makeCons(changes=[]):
    dict = {"s": 1, "pmax":1, "KCO_2": 1, "d":0.05, "umax" : 0.07, "DIC": 2, "KE": 0.2,"QHmin":0.1125, "QHmax":0.15, "QEmin": 0.03, "mE": 0.3, "mH": 0.03}

    for change in changes:
        dict[change[0]] = change[1]
    return dict


def makeFuncs(t, y, cD, rhoDOC, rhoDON):  
    E, H, QE, QH, C = y

    eH  = cD["s"]*(1-cD["QHmin"]/QH) 
    eE  = (1-cD["QEmin"]/QE)

    mH = cD["mH"]
    mE = cD["mE"]

    rhoCO2 = cD["pmax"] * C/(cD["KCO_2"]+C)
    rhoPhoto = (1-eE)*rhoCO2*E/H

    pH = eH*(rhoDOC(t,y,cD)+rhoPhoto)
    pE = eE*rhoCO2

    rhoResp = (1-eH)*(rhoDOC(t,y,cD)+rhoPhoto)*cD["DIC"]
                                             
    uH = rhoDON(t,y,cD) * (1-((QH-0.1125)/(0.15-0.1125))**2) 
    uE = cD["umax"] *(QH-cD["QHmin"])/( cD["KN"] + (QH-cD["QHmin"]) )           
    
    return pH, pE, mH, mE, uH, uE, rhoResp, rhoCO2   


def endo(t, y, cD, rhoDOC,rhoDON):
    E, H, QE, QH, C = y
    pH, pE, mH, mE, uH, uE, rhoResp, rhoCO2  = makeFuncs(t,y,cD,rhoDOC,rhoDON)

    dE = pE*E - mE*E 
    dH = pH*H - mH*H

    dQE = uE - pE*QE           
    dQH = uH - uE*E/H - pH*QH

    dC  = rhoResp - rhoCO2*E/H - cD["d"]*H**(-1/3)*C

    return [dE,dH,dQE,dQH,dC]


def _plotLimFac(sol, cD, xSpan, rhoDOC, rhoDON):
    E, H, QE, QH, C = sol.y
    pH, pE, mH, mE, uH, uE, rhoResp, rhoCO2  = makeFuncs(sol.t,sol.y,cD,rhoDOC,rhoDON)
    fig, (ax1,ax2,ax3) = plt.subplots(nrows=3, ncols=1)

    ax1.plot(sol.t,uE,"g--", label=r"$u_E$")
    ax1.plot(sol.t,uE*sol.y[0]/sol.y[1],"g", label=r"$u_E\frac{E}{H}$")
    ax1.plot(sol.t,uH,"b--", label="$u_H$")
    ax1.plot(sol.t,pH*sol.y[3],"b", label=r"$\mu_HQ_H$")

    ax2.plot(sol.t,pH,"b", label=r"$\mu_H$")
    ax2.plot(sol.t,mH*np.ones(len(sol.t)), "r--", label="$m_H$")
    twin2 = ax2.twinx()
    twin2.plot(sol.t,mE*np.ones(len(sol.t)),"y--", label="$m_E$")
    twin2.plot(sol.t,pE,"g", label="$p_E$")

    ax3.plot(sol.t, rhoResp,"b", label=r"$\rho_{resp}$")
    ax3.plot(sol.t, rhoCO2,"g", label=r"$\rho_{CO_2}$")

    ax1.set_title("") 
    ax1.set_ylabel(r"days$^{-1}$")
    ax2.set_xlabel("days")
    ax1.set_xlim(xSpan)
    ax2.set_xlim(xSpan)
    ax3.set_xlim(xSpan)
    ax1.legend()
    ax2.legend()
    twin2.legend()
    ax3.legend()
  

def EtoHDiv(t,y,cD,rhoDOC,rhoDON):
    return y[0]/y[1] - 1
EtoHDiv.terminal = True

def extinctE(t,y,cD,rhoDOC,rhoDON):
    return y[0]-1e-10
extinctE.terminal = True


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.integrate as integ
    import scipy.signal as signal

    def rhoDOC(t,y,cD):
        E, H, QE, QH, C = y
        return 0.03 *4* (1-H/166)
    def rhoDON(t,y,cD):
        E, H, QE, QH, C = y
        return rhoDOC(t,y,cD)*0.15
    
    y0 = [1, 60, 0.04, 0.12, 4]
    tEnd = 600
    cD = makeCons([("s",1),("pmax",1), ("KCO_2", 0.4),("KN",0.03),("umax", 0.05),("mE",0.4),("mH",0.03)])
    sol = integ.solve_ivp(endo, y0=y0, t_span=[0,tEnd], args=(cD,rhoDOC,rhoDON), dense_output=False, method="Radau", max_step = np.inf, rtol=1e-8, atol = 1e-8, events=[EtoHDiv,extinctE])
    print(sol)


    fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1)
    if sol.t[-1]==tEnd:
        xSpan = [-5,tEnd]
    else:
        xSpan = [0,sol.t[-1]]
    

    ax1.semilogy(sol.t,sol.y[0],"C2",label="E")
    ax1.semilogy(sol.t,sol.y[1],"C0",label="H")
    ax1.semilogy(sol.t,sol.y[4],"k",label="C")

    #hTest = y0[1]*np.exp(-cD["mH"]*sol.t)    
    #ax1.semilogy(sol.t,hTest,"b--",label="test H")
    
   
    ax2.plot(sol.t,sol.y[2],"C2", label = "$Q_E$")
    ax2.plot(sol.t,sol.y[3],"C0", label = "$Q_H$")
    ax2.plot(sol.t,sol.y[0]/sol.y[1],"gold", label = "$E/H$")

    #ax1.set_title(r"$\rho_{max}$ =" + f"{cD["rho"]}, $Q_F$ = {cD["QFood"]} ") 
    ax1.set_ylabel(r"mol C /m$^2$")
    ax2.set_ylabel("molar ratio")
    ax2.set_xlabel("d")
    ax1.set_xlim(xSpan)
    ax2.set_xlim(xSpan)
    if sol.t_events[0].size>0:
        ax2.set_ylim([None,0.5])
    ax1.legend()
    ax2.legend()

    _plotLimFac(sol,cD,xSpan,rhoDOC,rhoDON)
    plt.show()