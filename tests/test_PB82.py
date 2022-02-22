import numpy as np
from matplotlib import pyplot as plt
import pytzer as pz

temperature = np.linspace(0, 350, num=1000)
ln_kCO2 = pz.dissociation.CO2_PB82(T=temperature + 273.15)
ln_kH2CO3_PB82 = pz.dissociation.H2CO3_PB82(T=temperature + 273.15)
ln_kHCO3_PB82 = pz.dissociation.HCO3_PB82(T=temperature + 273.15)
ln_kH2CO3_MP98 = pz.dissociation.H2CO3_MP98(T=temperature + 273.15)
ln_kHCO3_MP98 = pz.dissociation.HCO3_MP98(T=temperature + 273.15)
ln10 = np.log(10)

fig, axs = plt.subplots(nrows=2, ncols=2, dpi=300)
# for ax in axs.ravel():
#     ax.set_xlim([0, 50])
ax = axs[0, 0]
ax.plot(temperature, ln_kCO2 / ln10)
ax.set_xlabel("Temperature / °C")
ax.set_ylabel("log$_{10}$ $K_\mathrm{H}$")
ax = axs[0, 1]
ax.plot(temperature, ln_kH2CO3_PB82 / ln10)
# ax.plot(temperature, ln_kH2CO3_MP98 / ln10)
ax.set_xlabel("Temperature / °C")
ax.set_ylabel("log$_{10}$ $K_1$")
ax = axs[1, 0]
ax.plot(temperature, ln_kHCO3_PB82 / ln10)
# ax.plot(temperature, ln_kHCO3_MP98 / ln10)
ax.set_xlabel("Temperature / °C")
ax.set_ylabel("log$_{10}$ $K_2$")
plt.tight_layout()
