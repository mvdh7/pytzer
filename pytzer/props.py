from autograd.numpy import array, float_

##### IONIC CHARGES ###########################################################

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
    
    return array([z[ion] for ion in ions])
