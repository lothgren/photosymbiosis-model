## I dont even know 1/8
import matplotlib.pyplot as plt
import numpy as np


def makeCons(changes=[]):
    dict = {"s": 1, "umax" : 0.07, "KL": 0.2, "KN": 0.2, "b": 1,"QHmin":0.1125, "QHmax":0.15, "QEmin": 0.03, "mE": 0.3, "mH": 0.03}

    for change in changes:
        dict[change[0]] = change[1]
    return dict


def makeFuncs(t, y, cD, rhoDOC,rhoDON):  
    L, H, QE, QH = y[0], y[1], y[2], y[3]

    eH  = cD["s"]*(1-cD["QHmin"]/QH) 
    eE  = (1-cD["QEmin"]/QE)
    g = cD["b"]*(1-eE)*(1-eH)

    n = g*L/(cD["KL"]+(1-g)*L)

    mH = cD["mH"]
    mE = cD["mE"]

    rhoFood  = rhoDOC(t,y,cD)         
    rhoResp  = cD["b"]*(1-eH)*(1+n)*rhoFood*1/(cD["KL"]+L)
    rhoPhoto = n*(rhoFood)

    pH = eH*(1+n)*rhoFood
    pE = eE*rhoResp
                                             
    uH = rhoDON(t,y,cD) * (1-((QH-0.1125)/(0.15-0.1125))**2) 
    uE = cD["umax"] *(QH-cD["QHmin"])/( cD["KN"] + (QH-cD["QHmin"]) )           
    
    return pH, pE, mH, mE, uH, uE, n, rhoFood, rhoResp, rhoPhoto   


def endo(t, y, cD, rhoDOC, rhoDON):
    L, H, QE, QH = y
    pH, pE, mH, mE, uH, uE, n, rhoFood, rhoResp, rhoPhoto = makeFuncs(t,y,cD,rhoDOC,rhoDON)

    dL = pE - mE*L - pH*L + mH*L
    dH = pH*H - mH*H

    dQE = uE - pE*QE/L           
    dQH = uH - uE*L - pH*QH 

    return [dL,dH,dQE,dQH]


def _plotLimFac(sol, cD, xSpan, rhoDOC, rhoDON):
    L, H, QE, QH = sol.y
    pH, pE, mH, mE, uH, uE, n, rhoFood, rhoResp, rhoPhoto = makeFuncs(sol.t,sol.y,cD,rhoDOC,rhoDON)
    fig, (ax1,ax2,ax3) = plt.subplots(nrows=3, ncols=1)

    ax1.plot(sol.t,uE,"g--", label=r"$u_E$")
    ax1.plot(sol.t,uE*L,"g", label=r"$u_E\frac{E}{H}$")
    ax1.plot(sol.t,uH,"b--", label="$u_H$")
    ax1.plot(sol.t,pH*sol.y[3],"b", label=r"$\mu_HQ_H$")

    
    ax2.plot(sol.t,pH,"b", label=r"$\mu_H$")
    ax2.plot(sol.t,mH*np.ones(len(sol.t)), "r--", label="$m_H$")
    ax2.plot(sol.t, rhoPhoto,"g", label=r"$\rho_{photo}$")
    ax2.plot(sol.t, rhoFood,"k", label=r"$\rho_{Food}$")

    ax3.plot(sol.t, rhoResp,"b", label=r"$\rho_{resp}$")
    ax3.plot(sol.t,pH+mE*np.ones(len(sol.t)),"r--", label="$m_E$")
    ax3.plot(sol.t,pE/L+mH,"g", label="$p_E$")

    ax1.set_title("") 
    ax1.set_ylabel(r"days$^{-1}$")
    ax2.set_xlabel("days")
    ax1.set_xlim(xSpan)
    ax2.set_xlim(xSpan)
    ax3.set_xlim(xSpan)
    ax1.legend()
    ax2.legend()
    ax3.legend()
    
def _plotFitness(sol, cD, xSpant, rhoDOC, rhoDON):
    pH, pE, mH, mE, uH, uE, n, rhoFood, rhoResp, rhoPhoto = makeFuncs(sol.t,sol.y,cD,rhoDOC,rhoDON)
    fig, ax =plt.figure(), plt.subplot()

    ax.plot(sol.t, pH-mH, "C0", label="H fitness")
    ax.plot(sol.t, pE-mE, "C2", label="E fitness")
    ax.set_title("Fitness of the partners") 
    ax.set_ylabel(r"days$^{-1}$")
    ax.set_xlabel("days")
    ax.set_xlim(xSpan)
    ax.legend()


def EtoHDiv(t,y,cD,rhoDOC,rhoDON):
    return y[0] - 1
EtoHDiv.terminal = True

def extinctE(t,y,cD,rhoDOC,rhoDON):
    return y[0]*y[1]-1e-10
extinctE.terminal = True


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.integrate as integ
    import scipy.signal as signal

    def rhoDOC(t,y,cD):
        E, H, QE, QH = y[0], y[1], y[2], y[3]
        return 0.03 *4* (1-H/166)
    def rhoDON(t,y,cD):
        E, H, QE, QH = y[0], y[1], y[2], y[3]
        return rhoDOC(t,y,cD) * 0.15 + cD["mH"]*0
        
    y0 = [0.01, 60, 0.04, 0.12]
    tEnd = 600
    cD = makeCons([("s",1),("b",1),("KN",0.05),("KL",0.02),("umax", 0.07),("mE",0.4),("mH",0.03),("QHmin",0.1125),("QEmin",0.03)])
    sol = integ.solve_ivp(endo, y0=y0, t_span=[0,tEnd], args=(cD,rhoDOC,rhoDON), dense_output=False, method="Radau", max_step = np.inf, rtol=1e-8, atol = 1e-8, events=[EtoHDiv,extinctE])
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
    
   
    ax2.plot(sol.t,sol.y[2],"C2", label = "$Q_E$")
    ax2.plot(sol.t,sol.y[3],"C0", label = "$Q_H$")
    ax2.plot(sol.t,sol.y[0],"gold", label = "$E/H$")

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
    #_plotFitness(sol,cD,xSpan,rhoDOC,rhoDON)
    plt.show()