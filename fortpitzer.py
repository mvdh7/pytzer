import pytzer as pz

tp = (298.15, 10.1325)
bC_Na_Cl = pz.coefficients.bC_Na_Cl_M88(*tp)
bC_H_Cl = pz.coefficients.bC_H_Cl_CMR93(*tp)
theta_H_Na = pz.coefficients.theta_H_Na_MarChemSpec25(*tp)
psi_H_Na_Cl = pz.coefficients.psi_H_Na_Cl_HMW84(*tp)
