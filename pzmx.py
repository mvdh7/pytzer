from copy import deepcopy
from autograd import numpy as np
from autograd import elementwise_grad as egrad
from autograd import jacobian
import pytzer as pz

# Define new values
whichint = 'Na-Cl'
b0 =  0.08063
b1 =  0.2631
b2 =  0.0
C0 =  0.0002624
C1 = -0.01005
a1 =  2.0
a2 = -9.0
om =  2.5
newbC = np.array([b0, b1, b2, C0, C1, a1, a2, om])

# Import data
filename = 'testfiles/pzmx.csv'
mols, ions, tempK, pres = pz.io.getmols(filename)
cflib = deepcopy(pz.cflibs.MarChemSpec)
cflib.add_zeros(ions) # just in case

# Set up propagation functions
def osmdir(whichint, newbC, mols, ions, tempK, pres, cflib):
    cflib.bC[whichint] = lambda tempK, pres: (*newbC, True)
    return pz.model.osm(mols, ions, tempK, pres, cflib)
osmdirg = egrad(osmdir, argnum=1)
osmjac = jacobian(osmdir, argnum=1)

# Do osmotic coefficient calculations
osm = pz.model.osm(mols, ions, tempK, pres, pz.cflibs.MarChemSpec)
newargs = (whichint, newbC, mols, ions, tempK, pres, cflib)
osmnew = osmdir(*newargs)
osmg = osmdirg(*newargs)
osmj = osmjac(*newargs)
