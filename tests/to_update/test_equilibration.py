import pytzer as pz

# Select parameter library
pz.set_library(pz, "CWTD23")

totals = pz.get_totals()
totals.update(
    {
        "CO2": 0.002,
        "Na": 0.5023,
        "K": 0.081,
        "Cl": 0.6,
        "BOH3": 0.0004,
        "SO4": 0.02,
        "F": 0.001,
        "Sr": 0.02,
        "Mg": 0.01,
        "Ca": 0.05,
        "Br": 0.1,
    }
)


scr = pz.solve(totals, 298.15, 10.1325)
solutes = pz.totals_to_solutes(totals, scr.stoich, scr.thermo)
