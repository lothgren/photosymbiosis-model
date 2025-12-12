## Model ODEs for endosymbiosis model
## Version 3.01, includes carbon pool and assumes fixed cell-quota for host and a dynamic DIN pool
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

H, E, N, C = sp.symbols("H E N C", real=True)
s, to, b, pmax, KCO2, delC, CI, mE, mH, rho0, HCap, delN, NI, KNE, uEmax, KNH, uHmax, QE, QH, QFood, eps = sp.symbols(
    r"s TO \beta p_{\max} K_{CO_2} \delta_C C_I m_E m_H rho_{Food\,\max} H_{\max} \delta_N N_I K_E u_{E\,\max} K_H u_{H\,\max} Q_E Q_H Q_{food} \epsilon", real=True)


def makeCons(changes=[]):
    paraValues = {s: 1, to: 1, b: 0.5, pmax: 0.45, KCO2: 0.02, delC: 0.5, CI: 0.11, mE: 0.03, mH: 0.03, rho0: 0.07, HCap: 166, 
                  delN: 0.5, NI:0.001, KNE: 0.02, uEmax : 0.035, KNH: 0.0001, uHmax : 0.0045, QE: 0.15, QH: 0.15, QFood: 0.15, eps: 0}

    #paraValues = {s: 1, to: 1, b: 0.5, pmax: 0.7, KCO2: 0.01, delC: 0.5, CI: 0.1, mE: 0.1, mH: 0.03, rho0: 0.09, HCap: 166, 
    #              delN: 0.2, NI:0.005, KNE: 0.04, uEmax : 0.05, KNH: 0.01, uHmax : 0.01, QE: 0.15, QH: 0.15, QFood: 0.15}
    for change in changes:
        paraValues[change[0]] = change[1]
    return paraValues


def minApprox(a,b,e=1e-2):
    return ( a+b - ((a-b)**2+e)**(1/2) )/2


def makeFuncs(t, y, cD, minFunc = np.minimum):  
    H, E, N, C = y
    rhoDOC = cD[rho0]*(1-H/cD[HCap])  #np.maximum(0,cD["rho0"]*(1-H/cD["HCap"]))
    rhoDON = rhoDOC*cD[QFood]

    uH = cD[uHmax] *(N)/( cD[KNH] + N )
    uE = cD[uEmax] *(N)/( cD[KNE] + N )

    pE = cD[pmax] * C/(cD[KCO2] + C)
    nE = (uE/pE)/cD[QE]
    eE  = minFunc(1/(cD[s]+1), nE)

    rhoPhoto = (1-(1+cD[s])*eE)*pE*E/H
    nH = ( (rhoDON + uH)/(rhoDOC+rhoPhoto) )/cD[QH]
    eH  = minFunc(1/(cD[s]+1), nH)

    rH = cD[b]* ( cD[s]*eH*(rhoDOC + rhoPhoto) + cD[mH]*cD[to] )    #
    rE = cD[b]* ( cD[s]*eE*pE + cD[mE]*cD[to] )                     # Total respired CO2 from H and E

    lH = (1-eH/nH)*(rhoDON + uH)      #
    lE = (1-eE/nE)*uE                 # Nutrients leaking in from H and E

    netNH = -lH + uH
    netNE = - (lE - uE)*E/H

    muH = eH*(rhoDOC + rhoPhoto)
    muE = eE*pE

    return muH, muE, rhoDOC, rhoPhoto, pE, rH, rE, netNH, netNE, eH, eE
    

def endo(t, y, cD, minFunc = np.minimum):
    H, E, N, C = y
    muH, muE, rhoDOC, rhoPhoto, pE, rH, rE, netNH, netNE, eH, eE  = makeFuncs(t,y,cD,minFunc)

    dH = (muH-cD[mH]*cD[to])*H
    dE = (muE-cD[mE]*cD[to] - cD[eps])*E

    dN = cD[mH]*cD[to]*cD[QH] + cD[mE]*cD[to]*cD[QE]*E/H + cD[delN]*(cD[NI]-N) - netNH - netNE
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


    netH = (1-eH/nH)*(rhoDON + uH) - uH
    netE = eE/nE*uE                                   #uE if nLimE else eE*QE*pE

    muH = eH*(rhoDOC + rhoPhoto)
    muE = eE*pE

    dH = (muH-mH*to)*H
    dE = (muE-mE*to-eps)*E
    dN = mH*to*QH + mE*to*QE*E/H + netH - netE*E/H + delN*(NI-N)
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
    muH, muE, rhoDOC, rhoPhoto, pE, rH, rE, netNH, netNE, eH, eE  = makeFuncs(t,y,cD)

    #plt.style.use('tableau-colorblind10')
    fig, (ax1,ax2,ax3) = plt.subplots(nrows=3, ncols=1)

    ax1.plot(t,N,"C0", label = "$N$")
    twin1 = ax1.twinx()
    twin1.plot(t,C,"k--",label="C")

    #ax2.plot(t,uE,"g--", label=r"$u_E$")
    ax2.plot(t,netNE,"g", label=r"$(u_E-l_E)\frac{E}{H}$")
    ax2.plot(t, netNH,"b", label=r"$u_H-l_H$")
    ax2.plot(t,0*E+(cD[mH]*cD[QH]+cD[mE]*cD[QE]*E/H),"b--", label=r"$m_HQ_H+m_EQ_E\frac{E}{H}$")
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
    
    y0 = [25,0.1,0.0001,0.001]
    tEnd = 1500
    cD = makeCons([ (s,1.0),
        (pmax, 0.45), (uEmax,0.033), (uHmax,0.0045),

        (KCO2, 0.02), (KNE, 0.03),   (KNH, 0.0001),

        (NI, 0.001), (CI, 0.09)
                   ])   #[(mE,0.06),(uEmax,0.03),(KNE,0.01),(CI,0.13)])   
    cD2 = makeCons([(pmax, 0.7), (KCO2, 0.01), (delC, 0.5), (CI, 0.1), (mE, 0.1), (mH, 0.03), (rho0, 0.09), (HCap, 166), 
                  (delN, 0.2), (NI, 0.005), (KNE, 0.04), (uEmax, 0.05), (KNH, 0.01), (uHmax, 0.01)])

    sol = integ.solve_ivp(endo, y0=y0, t_span=[0,tEnd], args=(cD,), dense_output=False, vectorized=True, method="Radau", max_step = np.inf, rtol=1e-8, atol = 1e-8, events=[])
    print(sol)

    ### Plotting 
    H, E, N, C = sol.y
    muH, muE, rhoDOC, rhoPhoto, pE, rH, rE, netNH, netNE, eH, eE = makeFuncs(sol.t,sol.y,cD)
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
    twin1.set_ylabel("E biomass/H biomass")
    ax2.set_ylabel(r"d$^{-1}$")
    twin2.set_ylabel(r"d$^{-1}$")
    ax2.set_xlabel("d")


    ax1.legend()
    twin1.legend()
    ax2.legend(loc="upper right")
    twin2.legend(loc="lower right")

    _plotLimFac(t, sol.y, cD)
    plt.show()