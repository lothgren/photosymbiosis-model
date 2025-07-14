## Carbon recycling version with DIC coming in for endosymb 10/7
## Version 2.03
import matplotlib.pyplot as plt
import numpy as np


def makeCons(changes=[]):
    dict = {"s": 1, "b":1, "umax" : 0.01, "KN": 0.01, "KE": 1,"QHmin":0.1125, "QHmax":0.15, "QEmin": 0.03, "mE": 0.15, "mH": 0.02}

    for change in changes:
        dict[change[0]] = change[1]
    return dict


def makeFuncs(t, y, cD, rhoC, rhoDIC, rhoN):  
    E, H, QE, QH = y[0], y[1], y[2], y[3]

    eH  = cD["s"]*(1-cD["QHmin"]/QH) 
    eE  = (1-cD["QEmin"]/QE)
    g   = cD["b"]*(1-eE)*(1-eH)

    n = g*E/(cD["KE"] + (1-g)*E) 

    mH = cD["mH"]
    mE = cD["mE"]

    rhoFood  = rhoC(E,H,QE,QH)         
    rhoResp  = cD["b"]*(1+n)*( (1-eH)*rhoFood + rhoDIC(E,H,QE,QH) )* H/(cD["KE"]+E)
    rhoPhoto = n*(rhoFood + 1/(1-eH)*rhoDIC(E,H,QE,QH))

    pH = eH*(rhoFood+rhoPhoto)
    pE = eE*rhoResp
                                             
    uH = rhoN(E,H,QE,QH) 
    uE = cD["umax"] * H*(QH-cD["QHmin"])/( cD["KN"] + H*(QH-cD["QHmin"]) )           
    
    return pH, pE, mH, mE, uH, uE, n, rhoFood, rhoResp, rhoPhoto   


def endo(t, y, cD, rhoC, rhoDIC, rhoN):
    E, H, QE, QH = y[0], y[1], y[2], y[3]
    pH, pE, mH, mE, uH, uE, n, rhoFood, rhoResp, rhoPhoto = makeFuncs(t,y,cD,rhoC,rhoDIC,rhoN)

    dE = pE*E - mE*E 
    dH = pH*H - mH*H

    dQE = uE - pE*QE           
    dQH = uH - uE*E/H - pH*QH# + mH*QH *(1-(QH-cD["QHmin"])/(cD["QHmax"]-cD["QHmin"])) 

    return [dE,dH,dQE,dQH]


def _plotLimFac(sol, cD, xSpant, rhoC, rhoDIC, rhoN):
    pH, pE, mH, mE, uH, uE, n, rhoFood, rhoResp, rhoPhoto = makeFuncs(sol.t,sol.y,cD,rhoC,rhoDIC,rhoN)
    fig, (ax1,ax2,ax3) = plt.subplots(nrows=3, ncols=1)

    ax1.plot(sol.t,uE,"g--", label=r"$u_E$")
    ax1.plot(sol.t,uE*sol.y[0]/sol.y[1],"g", label=r"$u_E\frac{E}{H}$")
    ax1.plot(sol.t,uH+0*mH*sol.y[3] *(1-(sol.y[3]-cD["QHmin"])/(cD["QHmax"]-cD["QHmin"])),"b--", label="$u_H$")
    ax1.plot(sol.t,pH*sol.y[3],"b", label=r"$\mu_HQ_H$")

    
    ax2.plot(sol.t,pH,"b", label=r"$\mu_H$")
    ax2.plot(sol.t,mH*np.ones(len(sol.t)), "r--", label="$m_H$")
    ax2.plot(sol.t, rhoPhoto,"g", label=r"$\rho_{photo}$")
    ax2.plot(sol.t, rhoFood,"k", label=r"$\rho_{Food}$")

    ax3.plot(sol.t, rhoResp,"b", label=r"$\rho_{resp}$")
    ax3.plot(sol.t,mE*np.ones(len(sol.t)),"r--", label="$m_E$")
    #ax3.plot(sol.t,n,"k--",label=r"$\eta$")
    ax3.plot(sol.t,pE,"g", label="$p_E$")

    ax1.set_title("") 
    ax1.set_ylabel(r"days$^{-1}$")
    ax2.set_xlabel("days")
    ax1.set_xlim(xSpan)
    ax2.set_xlim(xSpan)
    ax1.legend()
    ax2.legend()
    ax3.legend()
    
    

def EtoHDiv(t,y,cD,rhoC,rhoDIC,rhoN):
    return y[0]/y[1] - 10
EtoHDiv.terminal = True

def extinctE(t,y,cD,rhoC,rhoDIC,rhoN):
    return y[0]-1e-10
extinctE.terminal = True


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.integrate as integ
    import scipy.signal as signal

    def rhoC(E,H,QE,QH):
        return 0.03 *2* (1-H/166)
    def rhoDIC(E,H,QE,QH):
        return 0.03
    def rhoN(E,H,QE,QH):
        return rhoC(E,H,QE,QH) * 0.15     # * (1-(QH-0.1125)/(0.15-0.1125))
        
    y0 = [1, 40, 0.04, 0.12]
    tEnd = 1500
    cD = makeCons([("s",1),("b",1),("KN",0.5),("KE",1),("umax", 0.01),("mE",0.15),("mH",0.03)])
    sol = integ.solve_ivp(endo, y0=y0, t_span=[0,tEnd], args=(cD,rhoC,rhoDIC,rhoN), dense_output=False, method="Radau", max_step = np.inf, rtol=1e-8, atol = 1e-8, events=[EtoHDiv,extinctE])
    print(sol)


    fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1)
    if sol.t[-1]==tEnd:
        xSpan = [-5,tEnd]
    else:
        xSpan = [0,sol.t[-1]]
    

    ax1.semilogy(sol.t,sol.y[0],"C2",label="E")
    ax1.semilogy(sol.t,sol.y[1],"C0",label="H")

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

    _plotLimFac(sol,cD,xSpan,rhoC,rhoDIC,rhoN)
    plt.show()