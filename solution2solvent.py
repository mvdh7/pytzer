import numpy as np

# Relative ionic masses in g/mol
ion2mass = {
    'H': 1.00794,
    'Li': 6.941,
    'Na': 22.98977,
    'K': 39.0983,
    'Rb': 85.4678,
    'Cs': 132.90545,
    'NH4': 18.03846,
    'Mg': 24.3050,
    'Ca': 40.078,
    'Sr': 87.62,
    'MgF': 43.3034,
    'CaF': 59.0764,
    'SrF': 106.6184,
    'Ba': 137.327,
    'TrisH': 122.14298,
    'MgOH': 41.31234,
    'Fejj': 55.845,
    'Fejjj': 55.845,
    'Cdjj': 112.411,
    'Ni': 58.6934,    
    'Cuj': 63.546,
    'Cujj': 63.546,
    'Zn': 65.409,
    'F': 18.99840,
    'Cl': 35.453,
    'Br': 79.904,
    'I': 126.90447,
    'NO3': 62.00490,
    'OH': 17.00734,
    'HSO4': 97.07054,
    'SO4': 96.06260,
    'HCO3': 61.01684,
    'CO3': 60.00890, 
    'BOH4': 78.84036,  
    'NH3': 17.03052,
    'CO2': 44.00950,
    'BOH3': 61.83302,
    'Tris': 121.13504,
    'HF': 20.00634, 
    'MgCO3': 83.3139,
    'CaCO3': 100.0869,
    'SrCO3': 147.6289,  
}

# Select which ionic mass to use for molinity to molality conversion of eles
ele2ionmass = {
    't_HSO4': 'SO4',
    't_trisH': 'tris',
    't_Mg': 'Mg',
    't_BOH3': 'BOH3',
    't_H2CO3': 'HCO3',
}

def solution2solvent(mols, ions, tots, eles):
    """Roughly convert molinity (mol/kg-solution) to molality (mol/kg-H2O)."""
    ionmasses = np.array([ion2mass[ion] for ion in ions])*mols.ravel()
    elemasses = (np.array([ion2mass[ele2ionmass[ele]] for ele in eles])*
        tots.ravel())
    totalsalts = (np.sum(ionmasses) + np.sum(elemasses))*1e-3 # kg
    u2v = 1 + totalsalts
    mols = mols*u2v
    tots = tots*u2v
    return mols, tots

mols = np.array([0.5, 0.5])
ions = np.array(['Na', 'Cl'])
tots = np.array([1.0])
eles = np.array(['t_HSO4'])
mols, tots = solution2solvent(mols, ions, tots, eles)
