import pytzer as pz
import numpy as np
from autograd import elementwise_grad as egrad

filename = 'testfiles/M88 Table 4.csv'
#delim = ','

mols,ions,T = pz.io.getmols(filename)

cf = pz.cfdicts.M88

osm = pz.model.osm(mols,ions,T,cf)

aw = pz.model.osm2aw(mols,osm)

acfs = pz.model.acfs(mols,ions,T,cf)

tx = np.array([1])

tJ = pz.jfuncs._Harvie_J(tx)

tJscipy = pz.jfuncs._Harvie_J_drv(tx)

fJegrad = egrad(pz.jfuncs._Harvie_J)

tJegrad = fJegrad(tx)
