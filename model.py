## Model ODEs for endosymbiosis model
import numpy as np
import sympy as sp

H, S, N, C = sp.symbols("H S N C", real=True)
eps, to, g, pmax, KCO2, delC, CI, mS, mH, rho0, HCap, delN, NI, KNS, uSmax, KNH, uHmax, QS, QH, QFood, iota = sp.symbols(
    r"\epsilon TO \gamma p_{\max} K_C d_C C_I m_S m_H f_{\max} H_{\max} d_N N_I K_S u_{S\,\max} K_H u_{H\,\max} Q_S Q_H Q_{food} \iota", real=True)


def make_cons(changes=[]):
    paraValues = {eps: 1, to: 1, g: 0.5, pmax: 0.5, KCO2: 0.02, delC: 0.5, CI: 0.09, mS: 0.03, mH: 0.03, rho0: 0.07, HCap: 120, 
                  delN: 0.5, NI:0.00009, KNS: 0.02, uSmax : 0.0375, KNH: 0.0002, uHmax : 0.0045, QS: 0.15, QH: 0.15, QFood: 0.15, iota: 0}
    for change in changes:
        paraValues[change[0]] = change[1]
    return paraValues


def min_approx(a,b,e=1e-2):
    return ( a+b - ((a-b)**2+e)**(1/2) )/2

def max_approx(a,b,e=1e-2):
    return ( a+b + ((a-b)**2+e)**(1/2) )/2


def make_funcs(t, y, cD, min_func = np.minimum, max_func = np.maximum):  
    H, S, N, C = y
    rhoDOC = max_func(0,cD[rho0]*(1-H/cD[HCap]))   #Maximum inclided to avoid spScial intitial values to produce un-ecological results
    rhoDON = rhoDOC*cD[QFood]

    uH = cD[uHmax] *(N)/( cD[KNH] + N )
    uS = cD[uSmax] *(N)/( cD[KNS] + N )

    pS = cD[pmax] * C/(cD[KCO2] + C)
    nS = (uS/pS)/cD[QS]
    eS  = min_func(1/(cD[eps]+1), nS)

    rhoPhoto = (1-(1+cD[eps])*eS)*pS*S/H
    nH = ( (rhoDON + uH)/(rhoDOC+rhoPhoto) )/cD[QH]
    eH  = min_func(1/(cD[eps]+1), nH)

    rH = cD[g]* ( cD[eps]*eH*(rhoDOC + rhoPhoto) + cD[mH]*cD[to] )    #
    rS = cD[g]* ( cD[eps]*eS*pS + cD[mS]*cD[to] )                     # Total respired CO2 from H and S

    lH = (1-eH/nH)*(rhoDON + uH)      #
    lS = (1-eS/nS)*uS                 # Nutrients leaking in from H and S

    netNH = -lH + uH
    netNS = - (lS - uS)*S/H

    muH = eH*(rhoDOC + rhoPhoto)
    muS = eS*pS

    return muH, muS, rhoDOC, rhoPhoto, pS, rH, rS, netNH, netNS, eH, eS
    

def endo(t, y, cD, min_func = np.minimum, max_func = np.maximum):
    H, S, N, C = y
    muH, muS, rhoDOC, rhoPhoto, pS, rH, rS, netNH, netNS, eH, eS  = make_funcs(t,y,cD,min_func,max_func)

    dH = (muH-cD[mH]*cD[to])*H
    dS = (muS-cD[mS]*cD[to] - cD[iota])*S

    dN = cD[mH]*cD[to]*cD[QH] + cD[mS]*cD[to]*cD[QS]*S/H + cD[delN]*(cD[NI]-N) - netNH - netNS - (muH-cD[mH])*N
    dC  = rH + rS*S/H - pS*S/H + cD[delC]*(cD[CI]-C) - (muH-cD[mH])*C

    return [dH,dS,dN,dC]


def endo_symbolic(nLimH = False, nLimS = True):
    rhoDOC = rho0*(1-H/HCap)
    rhoDON = rhoDOC*QFood

    uH = uHmax *(N)/( KNH + N )
    uS = uSmax *(N)/( KNS + N )

    pS = pmax * C/(KCO2 + C)
    nS = (uS/pS)/QS
    eS = nS if nLimS else 1/(1+eps)

    rhoPhoto = (1-(1+eps)*eS)*pS*S/H
    nH = ( (rhoDON + uH)/(rhoDOC+rhoPhoto) )/QH
    eH = nH if nLimH else 1/(1+eps)                     # sp.Min(1/(s+1), nH)

    rH = g* ( eps*eH*(rhoDOC + rhoPhoto) + mH*to )
    rS = g* ( eps*eS*pS + mS*to )


    netH = (1-eH/nH)*(rhoDON + uH) - uH
    netS = eS/nS*uS                                   # uS if nLimS else eS*QS*pS

    muH = eH*(rhoDOC + rhoPhoto)
    muS = eS*pS

    dH = (muH-mH*to)*H
    dS = (muS-mS*to-iota)*S
    dN = mH*to*QH + mS*to*QS*S/H + netH - netS*S/H + delN*(NI-N) - (muH-mH*to)*N
    dC  = rH + rS*S/H - pS*S/H + delC*(CI-C) - (muH-mH*to)*C

    return [dH,dS,dN,dC]


def symb_death(t,y,cD):
    return y[1]-1e-10
symb_death.terminal = True

def load_stop(t,y,cD):
    return 10-y[1]/y[0]
load_stop.terminal = True



if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.integrate as integ

    
    def _plot_lim_fac(t, y, cD):
        H, S,  N, C = y
        muH, muS, rhoDOC, rhoPhoto, pS, rH, rS, netNH, netNS, eH, eS  = make_funcs(t,y,cD)
        fig, (ax1,ax2,ax3) = plt.subplots(nrows=3, ncols=1)

        ax1.plot(t,N,"C0", label = "$N$")
        twin1 = ax1.twinx()
        twin1.plot(t,C,"k--",label="C")

        #ax2.plot(t,uS,"g--", label=r"$u_S$")
        ax2.plot(t,netNS,"g", label=r"$(u_S-l_S)\frac{S}{H}$")
        ax2.plot(t, netNH,"b", label=r"$u_H-l_H$")
        ax2.plot(t,0*S+(cD[mH]*cD[QH]+cD[mS]*cD[QS]*S/H),"b--", label=r"$m_HQ_H+m_SQ_S\frac{S}{H}$")
        ax2.plot(t,cD[delN]*(cD[NI]-N), "r--", label=r"$\delta_N (N_I-N)$")
    
        ax3.plot(t, rH,"b", label=r"$r_{H}$")
        ax3.plot(t, rS*S/H,"b--", label=r"$r_{S}\frac{S}{H}$")
        ax3.plot(t, pS*S/H,"g--", label=r"$p_{S}\frac{S}{H}$")
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
        plt.plot(t,eS,"C2")
    
    y0 = [5, 0.001, 1e-06, 1e-05]
    tEnd = 300
    cD = make_cons([])

    sol = integ.solve_ivp(endo, y0=y0, t_span=[0,tEnd], args=(cD,), dense_output=False, vectorized=True, method="Radau", max_step = np.inf, rtol=1e-8, atol = 1e-8, events=[])
    print(sol)

    ### Plotting 
    H, S, N, C = sol.y
    muH, muS, rhoDOC, rhoPhoto, pS, rH, rS, netNH, netNS, eH, eS = make_funcs(sol.t,sol.y,cD)
    t = sol.t

    fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1)
    
    if y0[1]!=0:
        ax1.semilogy(t,S,"C2",label="S")
    ax1.semilogy(t,H,"C0",label="H")
    twin1 = ax1.twinx()
    twin1.plot(t,S/H,"gold", label = "$S/H$")
    
    twin2 = ax2.twinx()
    twin2.plot(t,muS,"g", label=r"$\mu_{S}$")
    twin2.plot(t,pS,"g--", label=r"$p_{S}$")
    ax2.plot(t,muH,"b", label=r"$\mu_{H}$")
    ax2.plot(t,rhoPhoto,"r--", label=r"$\rho_{photo}$")
    ax2.plot(t,rhoDOC,"k--", label=r"$\rho_{Food}$")


    ax1.set_ylabel(r"mol C /m$^2$")
    twin1.set_ylabel("S biomass/H biomass")
    ax2.set_ylabel(r"d$^{-1}$")
    twin2.set_ylabel(r"d$^{-1}$")
    ax2.set_xlabel("d")


    ax1.legend()
    twin1.legend()
    ax2.legend(loc="upper right")
    twin2.legend(loc="lower right")

    _plot_lim_fac(t, sol.y, cD)
    plt.show()