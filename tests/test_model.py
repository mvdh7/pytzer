import pytzer as pz, numpy as np

molalities = np.array([0.1, 0.1]) * 60
charges = np.array([+1, -1])

T = 273
P = 10
ca = [[pz.parameters.bC_Na_Cl_A92ii(T, P)[:-1]]]

args = pz.split_molalities_charges(molalities, charges)
params = dict(Aphi=0.3763, ca=ca)

g = pz.Gibbs_nRT(*args, **params).item()
ln_acfs = pz.log_activity_coefficients(*args, **params)
acfs = pz.activity_coefficients(*args, **params)

ln_mean_acf = pz.log_activities_to_mean(ln_acfs[0], ln_acfs[1], 1, 1).item()
mean_acf = np.exp(ln_mean_acf)

osm = pz.osmotic_coefficient(*args, **params).item()
ln_aw = pz.log_activity_water(*args, **params).item()
aw = pz.activity_water(*args, **params).item()
