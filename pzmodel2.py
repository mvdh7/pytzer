import pytzer as pz
#import numpy as np

mols, ions, tempK = pz.io.getmols('testfiles/pytzerPitzer.csv')

cf = pz.cfdicts.MarChemSpec

# Run the original model
zs = pz.props.charges(ions)[0]
I = pz.model.Istr(mols,zs)

fG = pz.model.fG(tempK,I,cf)

Gex_nRT = pz.model.Gex_nRT(mols,ions,tempK,cf)
acfs = pz.model.acfs(mols,ions,tempK,cf)

osm = pz.model.osm(mols,ions,tempK,cf)
aw = pz.model.osm2aw(mols,osm)

## Run the new model
#mols2 = np.transpose(mols)
#tempK2 = tempK.ravel()
#
#zs2 = np.vstack(zs)
#I2 = pz.model2.Istr(mols2,zs2)
#
#fG2 = pz.model2.fG(tempK,I2,cf)
#
#Gex_nRT2 = pz.model2.Gex_nRT(mols2,ions,tempK2,cf)
#acfs2 = pz.model2.acfs(mols2,ions,tempK2,cf)
#
#aw2 = pz.model2.aw(mols2,ions,tempK2,cf)
#osm2 = pz.model2.osm(mols2,ions,tempK2,cf)


