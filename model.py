## Model ODEs for endosymbiosis model
## Version 3.01, includes carbon pool and assumes fixed cell-quota for host and a dynamic DIN pool
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

H, E, N, C = sp.symbols("H E N C", real=True)
s, to, b, pmax, KCO2, delC, CI, mE, mH, rho0, HCap, delN, NI, KNE, uEmax, KNH, uHmax, QE, QH, QFood = sp.symbols(
    r"s TO \beta p_{\max} K_{CO_2} d_C C_I m_E m_H rho_{Food\,\max} H_{\max} d_N N_I K_E u_{E\,\max} K_H u_{H\,\max} Q_E Q_H Q_{food}", real=True)


def makeCons(changes=[]):
    paraValues = {s: 1, to: 1, b: 0.5, pmax: 0.7, KCO2: 0.01, delC: 0.5, CI: 0.1, mE: 0.1, mH: 0.03, rho0: 0.03*3, HCap: 166, 
                  delN: 0.2, NI:0.005, KNE: 0.05, uEmax : 0.05, KNH: 0.01, uHmax : 0.04, QE: 0.15, QH: 0.15, QFood: 0.15}
    for change in changes:
        paraValues[change[0]] = change[1]
    return paraValues


#def makeCons(changes=[]):
#    dict = {"s": 1, "to": 1, "b": 0.5, "pmax": 0.7, "KCO_2": 0.01, "dC": 0.5, "CI": 0.1, "mE": 0.10, "mH": 0.03, "rho0": 0.03*3, "HCap": 166,
#            "dN": 0.2, "NI": 0.005, "KNE": 0.05, "uEmax" : 0.05, "KNH": 0.01, "uHmax" : 0.04, "QE": 0.15, "QH": 0.15, "QFood": 0.15}
#
#    for change in changes:
#        dict[change[0]] = change[1]
#    return dict


def minApprox(a,b,e=1e-4):
    return ( a+b - ((a-b)**2+e)**(1/2) )/2


def makeFuncs(t, y, cD):  
    H, E, N, C = y
    rhoDOC = cD[rho0]*(1-H/cD[HCap])  #np.maximum(0,cD["rho0"]*(1-H/cD["HCap"]))
    rhoDON = rhoDOC*cD[QFood]

    uH = cD[uHmax] *(N)/( cD[KNH] + N )
    uE = cD[uEmax] *(N)/( cD[KNE] + N )

    pE = cD[pmax] * C/(cD[KCO2] + C)
    nE = (uE/pE)/cD[QE]
    eE  = np.minimum(1/(cD[s]+1), nE)         #minApprox(1/(cD[s]+1), nE)

    rhoPhoto = (1-(1+cD[s])*eE)*pE*E/H
    nH = ( (rhoDON + uH)/(rhoDOC+rhoPhoto) )/cD[QH]
    eH  = np.minimum(1/(cD[s]+1), nH)         #minApprox(1/(cD[s]+1), nH)

    rH = cD[b]* ( cD[s]*eH*(rhoDOC + rhoPhoto) + cD[mH]*cD[to])
    rE = cD[b]* ( cD[s]*eE*pE + cD[mE]*cD[to] )

    rhoDIN = (1-eH/nH)*(rhoDON + uH) + cD[mH]*cD[to]*cD[QH] + cD[mE]*cD[to]*cD[QE]*E/H

    muH = eH*(rhoDOC + rhoPhoto)
    muE = eE*pE

    return rE, rH, pE, muH, muE, uH, uE, rhoPhoto, rhoDIN, rhoDOC, rhoDON, eH, eE


def endo(t, y, cD):
    H, E, N, C = y
    rE, rH, pE, muH, muE, uH, uE, rhoPhoto, rhoDIN, rhoDOC, rhoDON, eH, eE  = makeFuncs(t,y,cD)

    dH = (muH-cD[mH]*cD[to])*H
    dE = (muE-cD[mE]*cD[to])*E

    dN = rhoDIN - uH - uE*E/H + cD[delN]*(cD[NI]-N)
    dC  = rH + rE*E/H - pE*E/H + cD[delC]*(cD[CI]-C)

    return [dH,dE,dN,dC]


def endoSymbolic(nLimH = False, nLimE = True):
    rhoDOC = rho0*(1-H/HCap)
    rhoDON = rhoDOC*QFood

    uH = uHmax *(N)/( KNH + N )
    uE = uEmax *(N)/( KNE + N )

    pE = pmax * C/(KCO2 + C)
    nE = (uE/pE)/QE
    eE = nE if nLimE else 1/(1+s)

    rhoPhoto = (1-(1+s)*eE)*pE*E/H
    nH = ( (rhoDON + uH)/(rhoDOC+rhoPhoto) )/QH
    eH = nH if nLimH else 1/(1+s)                # sp.Min(1/(s+1), nH)

    rH = b* ( s*eH*(rhoDOC + rhoPhoto) + mH*to )
    rE = b* ( s*eE*pE + mE*to )

    rhoDIN = (1-eH/nH)*(rhoDON + uH) + mH*to*QH + mE*to*QE*E/H

    muH = eH*(rhoDOC + rhoPhoto)
    muE = eE*pE

    dH = (muH-mH*to)*H
    dE = (muE-mE*to)*E
    dN = rhoDIN - uH - uE*E/H + delN*(NI-N)
    dC  = rH + rE*E/H - pE*E/H + delC*(CI-C)

    return [dH,dE,dN,dC]




def symbDeath(t,y,cD):
    return y[1]-1e-10
symbDeath.terminal = True

def loadStop(t,y,cD):
    return 2-y[1]/y[0]
loadStop.terminal = True


def _plotLimFac(t, y, cD):
    H, E,  N, C = y
    rE, rH, pE, muH, muE, uH, uE, rhoPhoto, rhoDIN, rhoDOC, rhoDON, eH, eE  = makeFuncs(t,y,cD)

    fig, (ax1,ax2,ax3) = plt.subplots(nrows=3, ncols=1)

    ax1.plot(t,N,"C0", label = "$N$")
    twin1 = ax1.twinx()
    twin1.plot(t,C,"k--",label="C")

    ax2.plot(t,uE,"g--", label=r"$u_E$")
    ax2.plot(t,uE*E/H,"g", label=r"$u_E\frac{E}{H}$")
    ax2.plot(t,rhoDIN-uH,"b", label=r"$\rho_{DIN}-u_H$")
    ax2.plot(t,cD[delN]*(cD[NI]-N), "r--", label=r"$\delta_N (N_I-N)$")
 
    ax3.plot(t, rH,"b", label=r"$r_{H}$")
    ax3.plot(t, rE*E/H,"b--", label=r"$r_{E}\frac{E}{H}$")
    ax3.plot(t, pE*E/H,"g--", label=r"$p_{E}\frac{E}{H}$")
    ax3.plot(t, cD[delC]*(cD[CI]-C),"k--", label=r"$\delta_C (C_I-C)$")

    ax1.legend(loc="upper right")
    twin1.legend(loc="lower right")
    ax2.legend()
    ax3.legend()

    ax1.set_ylabel("mol N/mol C")
    twin1.set_ylabel("mol CO$_2$/mol C")
    ax2.set_ylabel("mol N/mol C/d")
    ax3.set_ylabel("mol C/mol C/d")
    ax3.set_xlabel("days")

    plt.figure()
    plt.plot(t,eH,"C0")
    plt.plot(t,eE,"C2")
    



if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.integrate as integ
    import scipy.signal as signal
    
    y0 = [125.597883597883, 47.9261178749686, 0.0214285714285714, 0.00749276352332382] 
    tEnd = 1500
    cD = makeCons([(uHmax,0.0011020408163265306),]) 
    sol = integ.solve_ivp(endo, y0=y0, t_span=[0,tEnd], args=(cD,), dense_output=False, vectorized=True, method="Radau", max_step = np.inf, rtol=1e-8, atol = 1e-8, events=[])
    print(sol)

    ### Plotting 
    H, E, N, C = sol.y
    rE, rH, pE, muH, muE, uH, uE, rhoPhoto, rhoDIN, rhoDOC, rhoDON, eH, eE = makeFuncs(sol.t,sol.y,cD)
    t = sol.t

    print(f"eH = {eH[1]}, eE = {eE[1]}")

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