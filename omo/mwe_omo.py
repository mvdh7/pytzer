from pyomo.environ import *

model = ConcreteModel()

model.x = Var( initialize=-1.2, bounds=(-50, 50) )

model.obj = Objective(
    expr= model.x - 5.,
    sense= minimize )
