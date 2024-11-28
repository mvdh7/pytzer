from .. import debyehueckel, unsymmetrical, parameters as p
from . import Library

library = Library()
library.update_Aphi(debyehueckel.Aosm_M88)
library.update_func_J(unsymmetrical.P75_eq47)
# Tables S14-S18 (beta and C coefficients)
library.update_ca("Ca", "Br", p.bC_Ca_Br_SP78)
library.update_ca("Ca", "BOH4", p.bC_Ca_BOH4_SRM87)
library.update_ca("Ca", "Cl", p.bC_Ca_Cl_GM89)  # CWTD23 cite M88 but uses GM89
library.update_ca(
    "Ca", "HCO3", p.bC_Ca_HCO3_CWTD23
)  # CWTD23 cite HM93 but it's not - it's POS85 with a digit missing
library.update_ca("Ca", "HSO4", p.bC_Ca_HSO4_HMW84)
library.update_ca("Ca", "OH", p.bC_Ca_OH_HMW84)
library.update_ca("Ca", "SO4", p.bC_Ca_SO4_HEW82)
library.update_ca("CaF", "Cl", p.bC_CaF_Cl_PM16)
library.update_ca("H", "Br", p.bC_H_Br_MP98)
library.update_ca("H", "Cl", p.bC_H_Cl_CMR93)
library.update_ca("H", "HSO4", p.bC_H_HSO4_CRP94)
library.update_ca("H", "SO4", p.bC_H_SO4_CRP94)
library.update_ca("K", "Br", p.bC_K_Br_CWTD23)
library.update_ca("K", "BOH4", p.bC_K_BOH4_CWTD23)  # CWTD23 cite SRRJ87 but it's not
library.update_ca("K", "Cl", p.bC_K_Cl_GM89)
library.update_ca("K", "CO3", p.bC_K_CO3_CWTD23)  # CWTD23 cite SRG87 but it's not
library.update_ca("K", "F", p.bC_K_F_CWTD23)  # CWTD cite PM73 + SP78
library.update_ca("K", "HCO3", p.bC_K_HCO3_RGW84)
library.update_ca("K", "HSO4", p.bC_K_HSO4_WM13)
library.update_ca("K", "OH", p.bC_K_OH_MP98)
library.update_ca("K", "SO4", p.bC_K_SO4_GM89)
library.update_ca("Mg", "Br", p.bC_Mg_Br_SP78)
library.update_ca("Mg", "BOH4", p.bC_Mg_BOH4_SRM87)  # CWTD23 cite 88SR, numbers agree
library.update_ca("Mg", "Cl", p.bC_Mg_Cl_PP87i)
library.update_ca("Mg", "HCO3", p.bC_Mg_HCO3_CWTD23)  # CWTD23 cite POS85 but it's not
library.update_ca("Mg", "HSO4", p.bC_Mg_HSO4_HMW84)
library.update_ca("Mg", "SO4", p.bC_Mg_SO4_PP86ii)
library.update_ca("MgF", "Cl", p.bC_MgF_Cl_PM16)
library.update_ca("MgOH", "Cl", p.bC_MgOH_Cl_HMW84)
library.update_ca("Na", "Br", p.bC_Na_Br_CWTD23)  # CWTD23 cite 73PM
library.update_ca("Na", "BOH4", p.bC_Na_BOH4_CWTD23)  # TODO check vs MP98 function
library.update_ca("Na", "Cl", p.bC_Na_Cl_M88)
library.update_ca(
    "Na", "CO3", p.bC_Na_CO3_CWTD23b
)  # TODO check code vs table (see functions)
library.update_ca("Na", "F", p.bC_Na_F_CWTD23)
library.update_ca(
    "Na", "HCO3", p.bC_Na_HCO3_CWTD23b
)  # TODO check code vs table (see functions)
library.update_ca("Na", "HSO4", p.bC_Na_HSO4_CWTD23)
library.update_ca("Na", "OH", p.bC_Na_OH_PP87i)
library.update_ca("Na", "SO4", p.bC_Na_SO4_M88)
library.update_ca("Sr", "Br", p.bC_Sr_Br_SP78)
library.update_ca("Sr", "BOH4", p.bC_Ca_BOH4_SRM87)  # CWTD23 use Ca function
library.update_ca("Sr", "Cl", p.bC_Sr_Cl_CWTD23)
library.update_ca("Sr", "HCO3", p.bC_Ca_HCO3_CWTD23)  # CWTD23 use Ca function
library.update_ca("Sr", "HSO4", p.bC_Ca_HSO4_HMW84)  # CWTD23 use Ca function
library.update_ca("Sr", "OH", p.bC_Ca_OH_HMW84)  # CWTD23 use Ca function
library.update_ca("Sr", "SO4", p.bC_Ca_SO4_HEW82)  # CWTD23 use Ca function

# Table S19 (cc theta and psi coefficients)
library.update_cc("Ca", "H", p.theta_Ca_H_RGO81)
library.update_cca("Ca", "H", "Cl", p.psi_Ca_H_Cl_RGO81)
library.update_cc("Ca", "K", p.theta_Ca_K_GM89)
library.update_cca("Ca", "K", "Cl", p.psi_Ca_K_Cl_GM89)
library.update_cc("Ca", "Mg", p.theta_Ca_Mg_HMW84)  # CWTD23 cite HW80
library.update_cca("Ca", "Mg", "Cl", p.psi_Ca_Mg_Cl_HMW84)  # CWTD23 cite HW80
library.update_cca("Ca", "Mg", "SO4", p.psi_Ca_Mg_SO4_HMW84)  # CWTD23 cite HEW82
library.update_cc("Ca", "Na", p.theta_Ca_Na_M88)
library.update_cca("Ca", "Na", "Cl", p.psi_Ca_Na_Cl_M88)
library.update_cca("Ca", "Na", "SO4", p.psi_Ca_Na_SO4_HMW84)
library.update_cc("H", "K", p.theta_H_K_HWT22)  # CWTD23 cite CMR93
library.update_cca("H", "K", "Br", p.psi_H_K_Br_PK74)
library.update_cca("H", "K", "Cl", p.psi_H_K_Cl_HMW84)
library.update_cca("H", "K", "SO4", p.psi_H_K_SO4_HMW84)
library.update_cca("H", "K", "HSO4", p.psi_H_K_HSO4_HMW84)
library.update_cc("H", "Mg", p.theta_H_Mg_RGB80)
library.update_cca("H", "Mg", "Cl", p.psi_H_Mg_Cl_RGB80)
library.update_cca("H", "Mg", "HSO4", p.psi_H_Mg_HSO4_HMW84)  # not cited, but in code
library.update_cc("H", "Na", p.theta_H_Na_HWT22)
library.update_cca("H", "Na", "Br", p.psi_H_Na_Br_PK74)
library.update_cca("H", "Na", "Cl", p.psi_H_Na_Cl_PK74)
library.update_cc("H", "Sr", p.theta_H_Sr_RGRG86)
library.update_cca("H", "Sr", "Cl", p.psi_H_Sr_Cl_RGRG86)
library.update_cca("K", "Mg", "Cl", p.psi_K_Mg_Cl_PP87ii)
library.update_cca("K", "Mg", "SO4", p.psi_K_Mg_SO4_HMW84)  # CWTD23 cite HW80
library.update_cca("K", "Mg", "HSO4", p.psi_K_Mg_HSO4_HMW84)
library.update_cc("K", "Na", p.theta_K_Na_GM89)
library.update_cca("K", "Na", "Br", p.psi_K_Na_Br_PK74)
library.update_cca("K", "Na", "Cl", p.psi_K_Na_Cl_GM89)
library.update_cca("K", "Na", "SO4", p.psi_K_Na_SO4_GM89)
library.update_cc("K", "Sr", p.theta_Na_Sr_MP98)  # CWTD23 use Na-Sr function
library.update_cca("K", "Sr", "Cl", p.psi_Na_Sr_Cl_MP98)  # CWTD23 use Na-Sr function
library.update_cca("Mg", "MgOH", "Cl", p.psi_Mg_MgOH_Cl_HMW84)
library.update_cc("Mg", "Na", p.theta_Mg_Na_HMW84)  # CWTD23 cite P75
library.update_cca("Mg", "Na", "Cl", p.psi_Mg_Na_Cl_PP87ii)
library.update_cca("Mg", "Na", "SO4", p.psi_Mg_Na_SO4_HMW84)  # CWTD23 cite HW80
library.update_cc("Na", "Sr", p.theta_Na_Sr_MP98)
library.update_cca("Na", "Sr", "Cl", p.psi_Na_Sr_Cl_MP98)

# Table S20 (aa theta and psi coefficients)
library.update_aa("BOH4", "Cl", p.theta_BOH4_Cl_CWTD23)
library.update_caa("Ca", "BOH4", "Cl", p.psi_Ca_BOH4_Cl_MP98)  # CWTD23 cite 02P
library.update_caa("Mg", "BOH4", "Cl", p.psi_Mg_BOH4_Cl_MP98)  # CWTD23 cite 02P
library.update_caa("Na", "BOH4", "Cl", p.psi_Na_BOH4_Cl_MP98)  # CWTD23 cite 02P
library.update_aa("BOH4", "SO4", p.theta_BOH4_SO4_FW86)
library.update_aa("Br", "OH", p.theta_Br_OH_PK74)
library.update_caa("K", "Br", "OH", p.psi_K_Br_OH_PK74)
library.update_caa("Na", "Br", "OH", p.psi_Na_Br_OH_PK74)
library.update_aa("CO3", "Cl", p.theta_CO3_Cl_PP82)
library.update_caa("Na", "CO3", "Cl", p.psi_Na_CO3_Cl_TM82)
library.update_aa("Cl", "F", p.theta_Cl_F_MP98)  # CWTD23 cite 88CB
library.update_caa("Na", "Cl", "F", p.psi_Na_Cl_F_CWTD23)  # CWTD23 cite 88CB
library.update_aa("Cl", "HCO3", p.theta_Cl_HCO3_PP82)
library.update_caa("Mg", "Cl", "HCO3", p.psi_Mg_Cl_HCO3_HMW84)
library.update_caa("Na", "Cl", "HCO3", p.psi_Na_Cl_HCO3_PP82)
library.update_aa("Cl", "HSO4", p.theta_Cl_HSO4_HMW84)
library.update_caa("H", "Cl", "HSO4", p.psi_H_Cl_HSO4_HMW84)
library.update_caa("Na", "Cl", "HSO4", p.psi_Na_Cl_HSO4_HMW84)
library.update_aa("Cl", "OH", p.theta_Cl_OH_CWTD23)  # CWTD23 cite 02P
library.update_caa("Ca", "Cl", "OH", p.psi_Ca_Cl_OH_HMW84)
library.update_caa("K", "Cl", "OH", p.psi_K_Cl_OH_HMW84)
library.update_caa("Na", "Cl", "OH", p.psi_Na_Cl_OH_PK74)
library.update_aa("Cl", "SO4", p.theta_Cl_SO4_M88)
library.update_caa("Ca", "Cl", "SO4", p.psi_Ca_Cl_SO4_HMW84)  # CWTD23 cite 82HE/80HW
library.update_caa("K", "Cl", "SO4", p.psi_K_Cl_SO4_GM89)  # CWTD23 cite M88
library.update_caa("Mg", "Cl", "SO4", p.psi_Mg_Cl_SO4_HMW84)  # CWTD23 cite 82HE/80HW
library.update_caa("Na", "Cl", "SO4", p.psi_Na_Cl_SO4_M88)
library.update_aa("CO3", "SO4", p.theta_CO3_SO4_HMW84)
library.update_caa("K", "CO3", "SO4", p.psi_K_CO3_SO4_HMW84)
library.update_caa("Na", "CO3", "SO4", p.psi_Na_CO3_SO4_HMW84)
library.update_aa("HCO3", "SO4", p.theta_HCO3_SO4_HMW84)
library.update_caa("Mg", "HCO3", "SO4", p.psi_Mg_HCO3_SO4_HMW84)
library.update_caa("Na", "HCO3", "SO4", p.psi_Na_HCO3_SO4_HMW84)
library.update_caa("K", "HSO4", "SO4", p.psi_K_HSO4_SO4_HMW84)
library.update_aa("OH", "SO4", p.theta_OH_SO4_HMW84)
library.update_caa("K", "OH", "SO4", p.psi_K_OH_SO4_HMW84)
library.update_caa("Na", "OH", "SO4", p.psi_Na_OH_SO4_HMW84)

# Table S21 (lambda and zeta coefficients)
library.update_na("BOH3", "Cl", p.lambd_BOH3_Cl_FW86)
library.update_nc("BOH3", "K", p.lambd_BOH3_K_FW86)
library.update_nc("BOH3", "Na", p.lambd_BOH3_Na_FW86)
library.update_nca("BOH3", "Na", "SO4", p.zeta_BOH3_Na_SO4_FW86)
library.update_na("BOH3", "SO4", p.lambd_BOH3_SO4_FW86)
library.update_nc("CO2", "Ca", p.lambd_CO2_Ca_HM93)
library.update_nca("CO2", "Ca", "Cl", p.zeta_CO2_Ca_Cl_HM93)
library.update_na("CO2", "Cl", p.lambd_CO2_Cl_HM93)
library.update_nca("CO2", "H", "Cl", p.zeta_CO2_H_Cl_HM93)
library.update_nc("CO2", "K", p.lambd_CO2_K_HM93)
library.update_nca("CO2", "K", "Cl", p.zeta_CO2_K_Cl_HM93)
library.update_nca("CO2", "K", "SO4", p.zeta_CO2_K_SO4_HM93)
library.update_nc("CO2", "Mg", p.lambd_CO2_Mg_HM93)
library.update_nca("CO2", "Mg", "Cl", p.zeta_CO2_Mg_Cl_HM93)
library.update_nca("CO2", "Mg", "SO4", p.zeta_CO2_Mg_SO4_HM93)
library.update_nc("CO2", "Na", p.lambd_CO2_Na_HM93)
library.update_nca("CO2", "Na", "Cl", p.zeta_CO2_Na_Cl_HM93)
library.update_nca("CO2", "Na", "SO4", p.zeta_CO2_Na_SO4_HM93)
library.update_na("CO2", "SO4", p.lambd_CO2_SO4_HM93)
library.update_nc("HF", "Na", p.lambd_HF_Na_MP98)  # CWTD23 cite 88CB
library.update_nc("MgCO3", "Na", p.lambd_MgCO3_Na_CWTD23)  # CWTD23 cite 83MT
