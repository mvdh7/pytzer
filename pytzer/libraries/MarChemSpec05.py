# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019  Matthew Paul Humphreys  (GNU GPLv3)
from .. import debyehueckel
# Waters and Millero (2013). Mar. Chem. 149, 8-22,
#  doi:10.1016/j.marchem.2012.11.003
from . import MarChemSpec25
name = 'MarChemSpec05'
dh = debyehueckel.Aosm_MarChemSpec05
jfunc = MarChemSpec25.jfunc
# Begin with MarChemSpec25 with Aosm fixed at 5 degC
# Add parameters from GT17 Supp. Info. Table S6 (simultaneous optimisation)
bC = MarChemSpec25.bC
theta = MarChemSpec25.theta
psi = MarChemSpec25.psi
lambd = MarChemSpec25.lambd
zeta = MarChemSpec25.zeta
mu = MarChemSpec25.mu
