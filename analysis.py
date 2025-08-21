#Doing all the simulations, last updated 13

from analysisTools import *
import sympy as sp

##### Functions and constants
## Functions describing environmental flow of carbon and nutrients

def rhoDOC(t,y,cD):
    E, H, QE, QH = y[0], y[1], y[2], y[3]
    return 0.03 *4* (1-H/166)
def rhoDIC(t,y,cD):
    E, H, QE, QH = y[0], y[1], y[2], y[3]
    return cD["mH"]*0.0
def rhoDON(t,y,cD):
    E, H, QE, QH = y[0], y[1], y[2], y[3]
    return rhoDOC(t,y,cD) * 0.15 + cD["mH"]*0.0


## Constants giving standard parameter calues

cons = [("s",1),("pmax",3), ("DIC",0.3),("KCO_2", 0.1),("KN",2),("umax", 0.07),("mE",0.15),("mH",0.03)]


###### Plotting bifurcation diagrams

paraList = [ ("umax", [0.015,0.17]), ("KN", [0.0,10]), ("pmax", [1,5]), ("KCO_2", [0.01,6]), ("mE", [0.037,0.4]),("mH", [0.0,0.04]) ]

for x in paraList:
    para, span = x
    plotBifur(para, span, [rhoDOC,rhoDON], cons)
    name = "figsCarbonPool/bifur_" + para + ".png"
    plt.savefig(name)

print("Done!")
    