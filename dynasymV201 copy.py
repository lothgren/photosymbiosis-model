## Fresh start 25/03
import matplotlib.pyplot as plt
import numpy as np


def makeCons(changes=[]):
    dict = {"pmax": 1, "umax" : 0.07, "KNE": 5, "emax": 0.25, "QHmax": 0.5, "QHmin": 0.1, "QFood": 0.2, "Qmin": 0.03, "mE": 0.03, "mH": 0.03, "rho": 0.012}

    for change in changes:
        dict[change[0]] = change[1]
    return dict


def makeFuncs(t, y, cD):
    E, H, QE, QH = y[0], y[1], y[2], y[3]

    rhoFood = cD["rho"]*(1-H/1000)         #np.sqrt(np.sin(np.pi/365*t)**2 )
                                             
    uE = cD["umax"] * H*(QH-cD["QHmin"])/( cD["KNE"] + H*(QH-cD["QHmin"]) ) 
    uH = rhoFood*cD["QFood"]                            

    mH =  cD["mH"]*(1 + (E/H/0.1))

    y = 0.9

    e       = cD["emax"]*(1-cD["QHmin"]/QH) 
    eE      = (1-cD["Qmin"]/QE)

    pEShare = cD["pmax"]*cD["Qmin"]/QE #* np.sqrt(np.sin(np.pi*t)**2 )                          #(1-eE)*y*(cD["pmax"] + (1-e)*rhoFood*H/E + 1.0*mH) / (1 - y*(1-eE)*(1-e))
    pE      = cD["pmax"]*(1-cD["Qmin"]/QE)  #* np.sqrt(np.sin(np.pi*t)**2 )                     #eE*y*(cD["pmax"] + (1-e)*(rhoFood+pEShare)*H/E + 1.0*mH*H/E)
         
    pH      = e*(rhoFood + pEShare*E/H)

    mE = cD["mE"]*(1+((E/H/0.1)**2) )  # experiment just a little more here

    
    return pH, pE, pEShare, e, mH, mE, uH, uE


def endo(t, y, cD):
    E, H, QE, QH = y[0], y[1], y[2], y[3]
    pH, pE, pEShare, e, mH, mE, uH, uE = makeFuncs(t,y,cD)

    dE = pE*E - mE*E 
    dH = pH*H - mH*H

    dQE = uE - pE*QE
    dQH = uH - uE*E/H - pH*QH + mH*QH* 0.9   #(1-(QH-cD["QHmin"])/(cD["QHmax"]-cD["QHmin"]))

    return [dE,dH,dQE,dQH]


def _plotLimFac(sol, cD, xSpan):
    pH, pE, pEShare, e, mH, mE, uH, uE = makeFuncs(sol.t,sol.y,cD)
    fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1)

    ax1.plot(sol.t,uE,"g--", label="$u_E$")
    ax1.plot(sol.t,uH,"b", label="$u_H$")
    twin1 = ax1.twinx()
    twin1.plot(sol.t,e, "gold", label="e")

    ax2.plot(sol.t,pE,"g--", label="$p_E$")
    ax2.plot(sol.t,pH,"b", label=r"$\mu_h$")
    ax2.plot(sol.t,e*pEShare*sol.y[0]/sol.y[1],"b--", label=r"$e\cdot \rho_{p}\frac{E}{H}$")
    #ax2.plot(sol.t, (1-e)/e*pH, "k--", label="resp CO2")
    
    ax1.set_title("") 
    ax1.set_ylabel(r"days$^{-1}$")
    ax2.set_xlabel("days")
    ax1.set_xlim(xSpan)
    ax2.set_xlim(xSpan)
    ax1.legend()
    ax2.legend()
    

def __paintCostBen(net,ax,t):
    signList = ((np.sign(net) == 1)*1)
    col = ["r","g"]
    xStart = 0
    vLineList = []
    for i in range(1,len(signList)):
        if signList[i-1] != signList[i]:
            xEnd = t[i]
            ax.axvspan(xStart, xEnd, facecolor=col[signList[i-1]], alpha=0.2, zorder=-100)
            vLineList.append(xEnd)
            xStart = xEnd
    ax.axvspan(xStart, t[-1], facecolor=col[signList[-1]], alpha=0.2, zorder=-100)
    return vLineList


def _plotCostBen(sol,cD,xSpan):
    uF, uH, uE, g, pF, pE, pEShare, pH, e, carbIn = makeFuncs(sol.y, cD)
    fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1)

    costH, benH = uE*sol.y[3]/sol.y[4]/cD["QHmin"] + cD["p"]*g*sol.y[2], e*pEShare*sol.y[3]/sol.y[4]
    costE, benE = pF, pE + g*sol.y[4]

    ax1.plot(sol.t,costH,"orange", label = "H costs")
    ax1.plot(sol.t,benH, "g", label = "H ben")
    ax2.plot(sol.t,costE,"orange", label = "E costs")
    ax2.plot(sol.t,benE, "g", label = "E ben")
    ax1.legend()
    ax2.legend()

    vList1 = __paintCostBen(net=benH-costH,ax=ax1,t=sol.t)
    vList2 = __paintCostBen(net=benE-costE,ax=ax2,t=sol.t)

    for time in vList1+vList2:
        ax1.axvline(x=time, color = "k", dashes = (2,2), alpha = 0.5, lw = 0.5)
        ax2.axvline(x=time, color = "k", dashes = (2,2), alpha = 0.5, lw = 0.5)
    ax1.axhline(y=0, color = "k", alpha = 0.7, lw = 0.5)
    ax2.axhline(y=0, color = "k", alpha = 0.7, lw = 0.5)

    ax1.set_xlim(xSpan)
    ax2.set_xlim(xSpan)
    ax2.set_ylim([None,5])

    fig.supylabel("fitness")
    ax2.set_xlabel("days")

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
    
    y0 = [0.04, 40, 0.1, 0.18]
    tEnd = 1000
    cD = makeCons([("pmax", 5),("KNE",10),("umax", 0.07),("rho",0.012),("QHmin",0.17),("QHmax",0.3),("QFood",0.17), ("Qmin",0.1),("emax",0.5)])

    sol = integ.solve_ivp(endo, y0=y0, t_span=[0,tEnd], args=(cD,), dense_output=False, method="Radau", max_step = np.inf, rtol=1e-8, atol = 1e-8, events=[EtoHDiv,extinctE])
    print(sol)


    fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1)
    if sol.t[-1]==tEnd:
        xSpan = [0,tEnd]
    else:
        xSpan = [0,sol.t[-1]]
    
    ax1.semilogy(sol.t,sol.y[0],"g--",label="E")
    ax1.semilogy(sol.t,sol.y[1],"b",label="H")
    
   
    ax2.plot(sol.t,sol.y[2],"g--", label = "$Q_E$")
    ax2.plot(sol.t,sol.y[3],"b", label = "$Q_H$")
    ax2.plot(sol.t,sol.y[0]/sol.y[1],"gold", label = "$E/H$")

    ax1.set_title("Feeding rate = {}, $Q_F$ = {} ".format(cD["rho"],cD["QFood"])) 
    ax1.set_ylabel(r"$\mu$mol C (or N)/L")
    ax2.set_ylabel("molar ratio")
    ax2.set_xlabel("days")
    ax1.set_xlim(xSpan)
    ax2.set_xlim(xSpan)
    if sol.t_events[0].size>0:
        ax2.set_ylim([None,0.5])
    ax1.legend()
    ax2.legend()

    _plotLimFac(sol,cD,xSpan)
    #_plotCostBen(sol,cD,xSpan)
    plt.show()