import numpy as np
import pytzer as pz

def BRBJBLR(BR, BJ, BLR, T):
    return (BR + (BJ*(298.15**3/3) - 298.15**2*BLR)*(1/T - 1/298.15) +
        (BJ/6)*(T**2 - 298.15**2))
#    Tr = 298.15
#    return (BR + BJ*(Tr**3/3 - Tr**2*BLR)*(1/T - 1/Tr)
#       + BJ*(T**2 - Tr**2)/6)

tempK = T = 100*np.pi
#BR = 0.1298; BJ = -0.00000946; BLR = 9.914 * 0.0001
#BT0 = BRBJBLR(BR, BJ, BLR, tempK)
#BR = 0.32; BJ = -0.0000259; BLR = 11.86 * 0.0001
#BT1 = BRBJBLR(BR, BJ, BLR,tempK)
#BR = 0.0041; BJ = (2) * 0.000000319; BLR = -9.44 * 0.00001
#Cphi = BRBJBLR(BR, BJ, BLR, tempK)
  
BT0 = 276.82478 - 0.0028131778 + (-7375.5443 + 0.3701254) / T - 49.35997 * np.log(T) + (0.10945106 + 0.0000071788733) * T + (-0.000040218506 - 5.8847404E-09) * T**2 + 11.931122 / (T - 227) + (2.4824963 - 0.004821741) / (647 - T)
BT1 = 462.86977 + (-10294.18) / T - 85.96058 * np.log(T) + 0.23905969 * T + (-0.00010795894) * T**2
Cphi = -16.686897 + 0.00040534778 + (453.64961 - 0.051714017) / T + 2.9680772 * np.log(T) + (-0.0065161667 - 0.00000105530373) * T + (0.000002376578 + 8.9893405E-10) * T**2 - 0.68923899 / (T - 227) - 0.081156286 / (647 - T)

pres = 10.10325
interaction = 'Na-OH'
i0, i1 = interaction.split('-')
MIAMI = pz.libraries.MIAMI
BT0pz, BT1pz, BT2pz, C0pz, _, _, _, _, _ = MIAMI.bC[interaction](tempK, pres)
Cphipz = C0pz*(2*np.sqrt(np.abs(
    pz.properties._ion2charge[i0]*pz.properties._ion2charge[i1])))
