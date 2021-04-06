import pytzer as pz

#
# totS = 1.0
# totR = 1.5
# eleS = 'NaCl'
# eleR = 'KCl'
# tempK = 298.15
# pres = 10.10325
# prmlib = pz.libraries.Seawater
#
# pz.measure.isopiestic.getosmS(totS, totR, eleS, eleR, tempK, pres, prmlib)

e2i = pz.properties._ele2ions2
e2ik = list(e2i.keys())
e2ikl = [k.lower().replace("(", "") for k in e2ik]
e2iz = zip(e2ikl, e2ik)
e2ik = [k for _, k in sorted(e2iz)]

with open("e2i.txt", "w") as f:
    f.write("_ele2ions = {\n")
    for k in e2ik:
        if len(e2i[k][0]) == 2:
            kstr = "    '{}': (('{}', '{}'), ({:1.0f}, {:1.0f})),\n"
        elif len(e2i[k][0]) == 1:
            kstr = "    '{}': (('{}',), ({:1.0f})),\n"
        f.write(kstr.format(k, *e2i[k][0], *e2i[k][1]))
    f.write("}")
