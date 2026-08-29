# Photosymbiosis model
This repositories contains the code used to analyze and visualize the model

The repository is structred as follows:

### model.py
Contains the ODE function for model of project, photo, as well as its symbolic version.

### analysisTools.py
Contains tools used to analyse the model, e.g. simulating, finding fixed points, creating bifurcation diagrams numerically and symbolically

### analysis.py
Script where the tools are used to analyze the system and create figures used in the research article

N.B. Symbolic calculation is often time consuming (and the code is probably not as optomised as it could be) and thus for initial bifurcation analys it is recommended to use the numerics based functions

N.B. The proper documentation, annotation and general readability of the code *might* be update in a later time, either in this repository or in future projects
