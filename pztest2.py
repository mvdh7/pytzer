from autograd import grad
import pytzer as pz

bC = pz.coeffs.Na_Cl_A92ii(298.15)
bC2 = pz.coeffs.Na_Cl_M88(298.15)

AH = pz.coeffs.AH_MPH(298.15)
AC = pz.coeffs.AC_MPH(298.15)[0]

dAH = grad(lambda T: pz.coeffs.AH_MPH(T))(298.15)
