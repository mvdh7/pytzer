import pytzer as pz

# mols, ions, tempK, pres, prmlib, Gex_nRT, osm, aw, acfs =  pz.blackbox('testfiles/pytzerQuickStart.csv')


mols, ions, tempK, pres = pz.io.getmols("testfiles/pytzerQuickStart.csv")
# Calculate water activity
aw_Seawater = pz.model.aw(mols, ions, tempK, pres)
# Calculate again, but with MarChemSpec parameter library
aw_MarChemSpec = pz.model.aw(mols, ions, tempK, pres, prmlib=pz.libraries.MarChemSpec)
# Now with M88 parameter library, padded with zeros for missing interactions
from copy import deepcopy

myM88 = deepcopy(pz.libraries.M88)
myM88.add_zeros(ions)
aw_myM88 = pz.model.aw(mols, ions, tempK, pres, prmlib=myM88)
