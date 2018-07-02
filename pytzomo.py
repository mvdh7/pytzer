from pyomo.environ import *
#from pyomo.bilevel import *

M = ConcreteModel()

M.x = Var(bounds=(0,None))
M.y = Var(bounds=(0,None))
M.o = Objective(expr=M.x - 4*M.y)

#M.sub = SubModel(fixed=M.x)
