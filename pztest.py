import pytzer as pz

filename = 'testfiles/M88 Table 4.csv'
delim = ','

mols, ions, T = pz.io.getmols(filename,delim)

cf = pz.cdicts.M88

osm = pz.model.osm(mols,ions,T,cf)

aw = pz.model.osm2aw(mols, osm)
