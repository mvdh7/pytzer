import pytzer as pz, numpy as np

molalities = np.array([0.1, 0.1])
charges = np.array([+1, -1])

m_cats = np.compress(charges > 0, molalities)
m_anis = np.compress(charges < 0, molalities)
m_neus = np.compress(charges == 0, molalities)
z_cats = np.compress(charges > 0, charges)
z_anis = np.compress(charges < 0, charges)


T = 273
P = 10
ca = [[pz.parameters.bC_Na_Cl_A92ii(T, P)[:-1]]]


args = (m_cats, m_anis, m_neus, z_cats, z_anis)
kwargs = dict(Aosm=0.3763, ca=ca)

g = pz.model.Gibbs_nRT(*args, **kwargs).item()
ln_acfs = pz.model.log_activity_coefficients(*args, **kwargs)
acfs = pz.model.activity_coefficients(*args, **kwargs)

ln_mean_acf = pz.model.ln_acf2ln_acf_MX(ln_acfs[0], ln_acfs[1], 1, 1).item()
mean_acf = np.exp(ln_mean_acf)
