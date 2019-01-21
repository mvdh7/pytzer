from autograd.numpy import array, float_

# Get charges for input ions

def charges(ions):

    z = {'Ba'  : float_(+2),
         'Ca'  : float_(+2),
         'H'   : float_(+1),
         'K'   : float_(+1),
         'Mg'  : float_(+2),
         'Na'  : float_(+1),
         'Zn'  : float_(+2),
         
         'Br'  : float_(-1),
         'Cl'  : float_(-1),
         'OH'  : float_(-1),
         'HSO4': float_(-1),
         'SO4' : float_(-2)}
    
    zs = array([z[ion] for ion in ions])
    
    cations = ions[zs > 0]
    anions  = ions[zs < 0]
    
    return zs, cations, anions
