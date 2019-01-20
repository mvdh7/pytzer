import pytzer as pz

filename = 'pztest.csv'
delim = ','

mols, ions, T = pz.io.getmols(filename, delim)

cf = pz.cdicts.GM89

osm = pz.model.osm(mols,ions,T,cf)
