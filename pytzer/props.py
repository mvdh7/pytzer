# pytzer: the Pitzer model for chemical speciation
# Copyright (C) 2019  Matthew Paul Humphreys  (GNU GPLv3)

from autograd.numpy import array, float_

# Get charges for input ions
def charges(ions):

    # Define dict of charges
    #   Order: neutrals, then cations, then anions,
    #          and alphabetical within each section.
    z = {'tris' : float_(0),

         'Ba'   : float_(+2),
         'Ca'   : float_(+2),
         'Cs'   : float_(+1),
         'H'    : float_(+1),
         'K'    : float_(+1),
         'Li'   : float_(+1),
         'Mg'   : float_(+2),
         'MgOH' : float_(+1),
         'Na'   : float_(+1),
         'Sr'   : float_(+2),
         'trisH': float_(+1),
         'Zn'   : float_(+2),

         'BOH4' : float_(-1),
         'Br'   : float_(-1),
         'Cl'   : float_(-1),
         'I'    : float_(-1),
         'F'    : float_(-1),
         'OH'   : float_(-1),
         'HSO4' : float_(-1),
         'SO4'  : float_(-2),
         'SO4'  : float_(-2)}

    # Extract charges from dict
    zs = array([z[ion] for ion in ions])

    # Get lists of cation, anion and neutral names
    cations  = ions[zs >  0]
    anions   = ions[zs <  0]
    neutrals = ions[zs == 0]

    return zs, cations, anions, neutrals
