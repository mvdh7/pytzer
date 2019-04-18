import pytzer as pz
from autograd.numpy import sqrt
from autograd.numpy import abs as np_abs

class Binary:
    
    def __init__(self, cation, anion):
        self.cation = cation
        self.anion = anion
        self.zcat, self.zani = pz.properties.charges([cation, anion])[0]
        def none(T, P):
            return 0
        self.b0 = none
        self.b1 = none
        self.b2 = none
        self.C0 = none
        self.C1 = none
        self.alph1 = -9
        self.alph2 = -9
        self.omega = -9
        self.valid = lambda T: T > 0

    def Cphi2C0(self):
        self.C0 = lambda T, P: self.Cphi(T, P) / (2 * 
            sqrt(np_abs(self.zcat*self.zani)))
        
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

test = Binary('Na', 'Cl')
test.b0 = lambda T: T * 3
