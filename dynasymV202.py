## Carbon recycling version
import matplotlib.pyplot as plt
import numpy as np


def makeCons(changes=[]):
    dict = {"umax" : 0.02, "KN": 0.001, "KE": 0.1, "QHmax": 0.17, "QHmin": 0.1275, "QFood": 0.17, "QEmin": 0.03, "mE": 0.2, "mH": 0.02, "rho": 0.048}

    for change in changes:
        dict[change[0]] = change[1]
    return dict


def makeFuncsSimp(t, y, cD):  #Simple version again to see if it is enough
    E, H, QE, QH = y[0], y[1], y[2], y[3]

    eH  = (1-cD["QHmin"]/QH) 
    eE = (1-cD["QEmin"]/QE)
    b  = E/(cD["KE"]+E)

    n = b*(1-eE)*(1-eH)/(1 - b*(1-eH)*(1-eE)) 

    rhoFood  = cD["rho"]*(1-H/1000)         
    rhoResp  = b*(1-eH)*(1+n)*rhoFood * H/E

    pH = eH*(1+n)*rhoFood
    pE = eE*rhoResp
                                             
    uH = rhoFood*cD["QFood"] 
    uE = cD["umax"] * (QH-cD["QHmin"])/( cD["KN"] + (QH-cD["QHmin"]) )                   

    mH = cD["mH"]
    mE = cD["mE"]
    
    return pH, pE, mH, mE, uH, uE, eH, eE, rhoResp


def makeFuncs(t, y, cD):
    E, H, QE, QH = y[0], y[1], y[2], y[3]

    eH  = (1-cD["QHmin"]/QH) 
    eE = (1-cD["QEmin"]/QE)
    b  = 1.0 * (E/H) / (0.01+(E/H))

    mH = cD["mH"]

    n = b*(1-eE)/(1 - b*(1-eH)*(1-eE)) #(1-eH)*

    rhoFood  = cD["rho"]*(1-H/1000)               #*np.sqrt(np.sin(np.pi/365*t)**2 )
    rhoResp  = b*( (1-eH)*(1+(1-eH)*n)*rhoFood + (1+(1-eH)*n)*mH)

    pH = eH*((1+(1-eH)*n)*rhoFood + n*mH)
    pE = eE*rhoResp*H/E
                                             
    uH = rhoFood*cD["QFood"] 
    uE = cD["umax"] * (QH-cD["QHmin"])/( cD["KN"] + (QH-cD["QHmin"]) )                   

    
    c = 1000
    mE = cD["mE"]#*np.exp(-c*(pH-mH))  #*1/(rhoResp*H/E)    #* np.exp(c*(mH-pH))/(0.1+np.exp(c*(mH-pH)))
    
    return pH, pE, mH, mE, uH, uE, eH, eE, rhoResp


def endo(t, y, cD):
    E, H, QE, QH = y[0], y[1], y[2], y[3]
    pH, pE, mH, mE, uH, uE, eH, eE, rhoResp = makeFuncsSimp(t,y,cD)

    dE = pE*E - mE*E 
    dH = pH*H - mH*H

    dQE = uE - pE*QE           
    dQH = uH - uE*E/H - pH*QH + mH*QH *(1-(QH-cD["QHmin"])/(cD["QHmax"]-cD["QHmin"]))

    return [dE,dH,dQE,dQH]


def _plotLimFac(sol, cD, xSpan):
    pH, pE, mH, mE, uH, uE, eH, eE,rhoResp = makeFuncsSimp(sol.t,sol.y,cD)
    fig, (ax1,ax2,ax3) = plt.subplots(nrows=3, ncols=1)

    ax1.plot(sol.t,uE,"g--", label=r"$u_E$")
    ax1.plot(sol.t,uE*sol.y[0]/sol.y[1],"g", label=r"$u_E\frac{E}{H}$")
    ax1.plot(sol.t,uH+mH*sol.y[3] *(1-(sol.y[3]-cD["QHmin"])/(cD["QHmax"]-cD["QHmin"])),"b--", label="$u_H$")
    ax1.plot(sol.t,pH*sol.y[3],"b", label=r"$\mu_HQ_H$")

    ax2.plot(sol.t,pE,"g", label="$p_E$")
    ax2.plot(sol.t,pH,"b", label=r"$\mu_H$")
    ax2.plot(sol.t,mE*np.ones(len(sol.t)),"r--", label="$m_E$")
    ax2.plot(sol.t,mH*np.ones(len(sol.t)), color="orange", label="$m_H$")

    ax3.plot(sol.t, rhoResp*sol.y[0]/sol.y[1],"b", label=r"$\rho_{resp}\frac{H}{E}$")
    #ax3.plot(sol.t, (1-eE)*pE/eE,"g", label=r"$\rho_{photo}$")
    
    twin3 = ax3.twinx()
    twin3.plot(sol.t,eH,"b--",label="$e_H$")
    twin3.plot(sol.t,eE,"g--",label="$e_E$")

    ax1.set_title("") 
    ax1.set_ylabel(r"days$^{-1}$")
    ax2.set_xlabel("days")
    ax1.set_xlim(xSpan)
    ax2.set_xlim(xSpan)
    ax1.legend()
    ax2.legend()
    ax3.legend(loc="lower left")
    twin3.legend(loc="upper right")
    

def EtoHDiv(t,y,cD):
    return y[0]/y[1] - 10
EtoHDiv.terminal = True

def extinctE(t,y,cD):
    return y[0]-1e-10
extinctE.terminal = True


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.integrate as integ
    import scipy.signal as signal
    
    y0 = [4, 40, 0.04, 0.16]
    tEnd = 1200
    cD = makeCons([("KN",0.01),("umax", 0.03),("rho",0.012*4),("mE",0.15),("mH",0.02),("QHmin",0.1125),("QHmax",0.15),("QEmin",0.03),("QFood",0.17)])

    sol = integ.solve_ivp(endo, y0=y0, t_span=[0,tEnd], args=(cD,), dense_output=False, method="Radau", max_step = np.inf, rtol=1e-8, atol = 1e-8, events=[EtoHDiv,extinctE])
    print(sol)


    fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1)
    if sol.t[-1]==tEnd:
        xSpan = [-5,tEnd]
    else:
        xSpan = [0,sol.t[-1]]
    

    ax1.semilogy(sol.t,sol.y[0],"C2",label="E")
    ax1.semilogy(sol.t,sol.y[1],"C0",label="H")

    #hTest = y0[1]*np.exp(-cD["mH"]*sol.t)    
    #ax1.semilogy(sol.t,hTest)
    
   
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

    _plotLimFac(sol,cD,xSpan)
    plt.show()