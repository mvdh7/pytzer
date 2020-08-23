def ca_none(T, P):
    """cation:anion --- no interaction effect."""
    return {
        "beta0": 0,
        "beta1": 0,
        'alpha1': 1,
        "beta2": 0,
        'alpha2': 1,
        'C0': 0,
        'C1': 0,
        'omega': 1,
        'valid': (T > 0) & (P > 0),
    }



class CationAnion:
    
    def __init__(self, cation, anion, func=ca_none):
        self.cation = cation
        self.anion = anion
        self.func = func
        
        
    def get_parameters(self, T=298.15, P=1.01325):
        return self.func(T, P)


    def get_BCfunc(self, I):
        return


class ParameterLibrary:
    pass



class CationAnionInteractions:
    
    def __init__(self, name=None):
        self.name = name
        self.interactions = {}
        
    def __setattr__(self, name, value):
        super().__setattr__(name, value)


    def add_interaction(self, cation, anion, func=ca_none):
        self.interactions[cation + ":" + anion] = func



ca = CationAnion("Na", "Cl")

test = CationAnionInteractions(name="trying")
