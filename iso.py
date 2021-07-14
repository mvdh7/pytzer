from collections import OrderedDict
import pandas as pd, numpy as np
import pytzer as pz

# Import isopiestic dataset
iso = pd.read_excel("datasets/iso.xlsx", skiprows=2)


# Define conversions from salt molalities to solute molalities
salt_to_solutes = {
    "BaCl2": dict(Ba=1, Cl=2),
    "CaCl2": dict(Ca=1, Cl=2),
    "CsCl": dict(Cs=1, Cl=1),
    "CuCl2": dict(Cujj=1, Cl=2),
    "CuSO4": dict(Cujj=1, SO4=1),
    "Eu(NO3)3": dict(Eujjj=1, NO3=3),
    "H2SO4": dict(H=2, SO4=1),  # need to enable equilibration!
    "H3BO3": dict(H=3, BO3=1),  # need to enable equilibration?
    "K2B4O7": dict(K=2, B4O7=1),
    "K2CO3": dict(K=2, CO3=1),
    "KCl": dict(K=1, Cl=1),
    "LiCl": dict(Li=1, Cl=1),
    "MgCl2": dict(Mg=1, Cl=2),
    "MgSO4": dict(Mg=1, SO4=1),
    "Na2Mg(SO4)2": dict(Na=2, Mg=1, SO4=2),
    "Na2SO4": dict(Na=2, SO4=1),
    "NaCl": dict(Na=1, Cl=1),
    "NaNO3": dict(Na=1, NO3=1),
    "NiSO4": dict(Ni=1, SO4=1),
    "SrCl2": dict(Sr=1, Cl=2),
    "glycerol": dict(glycerol=1),
    "sucrose": dict(sucrose=1),
    "urea": dict(urea=1),
    "(trisH)2SO4": dict(trisH=2, SO4=1),
    "tris": dict(tris=1),
    "trisHCl": dict(trisH=1, Cl=1),
}


class IsopiesticExperiment:
    def __init__(self, iso_row=None):
        if isinstance(iso_row, pd.Series):
            self.get_iso_data(iso_row)

    def __repr__(self):
        outstr = "Isopiestic experiment from {}.\n".format(self.source)
        outstr += (
            "Temperature = {temperature} K.  Pressure = {pressure} dbar.\n".format(
                temperature=self.temperature, pressure=self.pressure
            )
        )
        outstr += "{} cups, containing:" "".format(self.cups)
        for cup in range(self.cups):
            outstr += "\n- {}: ".format(cup)
            for i, salt in enumerate(self.salts[cup]):
                if i < len(self.salts[cup]) - 1:
                    outstr += "{}, ".format(salt)
                else:
                    outstr += "{}.".format(salt)
        return outstr

    def get_iso_data(self, iso_row):
        """Import data from one row of the iso.xlsx spreadsheet."""
        # Get columns containing molality data
        molcols = molcols = [c for c in iso_row.index if c.startswith("Unnamed")]
        raw_mols = iso_row[molcols]
        raw_mols = raw_mols[~pd.isnull(raw_mols)].to_numpy()
        # Get conditions and metadata
        self.source = iso_row.src
        self.temperature = iso_row.tempK
        self.pressure = iso_row.pres
        # Convert salt molalities to solute molalities
        self.salts = {}
        self.salt_names = {}
        self.salt_totals = {}
        self.solutes = {}
        self.all_solutes = []
        cup = 0
        for xi, x in enumerate(raw_mols):
            if isinstance(x, str):
                self.salts[cup] = x.split("-")
                self.salt_names[cup] = x
                self.salt_totals[cup] = raw_mols[xi - len(self.salts[cup]) : xi]
                self.solutes[cup] = OrderedDict()
                for si, s in enumerate(self.salts[cup]):
                    for k, v in salt_to_solutes[s].items():
                        if k not in self.solutes[cup]:
                            self.solutes[cup][k] = 0.0
                        if k not in self.all_solutes:
                            self.all_solutes.append(k)
                        self.solutes[cup][k] += (
                            v * raw_mols[xi - len(self.salts[cup]) + si]
                        )
                cup += 1
        self.cups = cup
        return self

    def get_parameters(self, library, verbose=True):
        """Evaluate the Pitzer model parameters."""
        self.params = library.get_parameters(
            solutes=self.all_solutes,
            temperature=self.temperature,
            pressure=self.pressure,
            verbose=verbose,
        )
        return self

    def get_activity_water(self, library=pz.libraries.Seawater):
        """Calculate the water activity in all cups --- they should all be equal."""
        if not hasattr(self, "params"):
            self.get_parameters(library, verbose=False)
        self.activity_water = np.full(self.cups, np.nan)
        for cup, solutes in self.solutes.items():
            self.activity_water[cup] = pz.activity_water(solutes, **self.params)
        self.activity_water_std = np.std(self.activity_water)
        return self

    def get_experiment(self, reference=None, library=pz.libraries.Seawater):
        """Calculate the osmotic coefficients based on one reference cup."""
        assert isinstance(reference, str)
        # Identify the reference cup and calculate its osmotic coefficient
        self.cup_ref = [k for k, v in self.salts.items() if v == [reference]][0]
        # # Evaluate Pitzer model parameters
        # if not hasattr(self, "params"):
        #     self.get_parameters(library, verbose=False)
        # Calculate osmotic coefficients with the Pitzer model
        self.osmotic_coefficients = {
            "experiment": np.full(self.cups, np.nan),
            "model": np.full(self.cups, np.nan),
        }
        for cup, solutes in self.solutes.items():
            params = library.get_parameters(
                solutes=solutes,
                temperature=self.temperature,
                pressure=self.pressure,
                verbose=False,
            )
            self.osmotic_coefficients["model"][cup] = pz.osmotic_coefficient(
                solutes, **params
            ).item()
        # Calculate non-reference osmotic coefficients from the experiment data
        self.activity_water_ref = pz.convert.osmotic_to_activity(
            np.array([m for m in self.solutes[self.cup_ref].values()]),
            self.osmotic_coefficients["model"][self.cup_ref],
        )
        for cup, solutes in self.solutes.items():
            if cup == self.cup_ref:
                continue  # leave the reference cup as NaN
            self.osmotic_coefficients["experiment"][
                cup
            ] = pz.convert.activity_to_osmotic(
                np.array([m for m in solutes.values()]), self.activity_water_ref
            )
        self.osmotic_coefficients["offset"] = (
            self.osmotic_coefficients["experiment"] - self.osmotic_coefficients["model"]
        )
        return self


# # Run an experiment
# row = 1655  # for testing
# ie = IsopiesticExperiment(iso.loc[row]).get_experiment(reference="NaCl")
# print(ie)

# Get rows containing a particular salt (doesn't work if it's in a mixture)
rows_with = (iso == "NaCl").any(axis=1) & (iso == "KCl").any(axis=1)
# Run a series of experiments
ies = {}
reference = "NaCl"
for row in iso.index[rows_with]:
    print(row)
    try:
        ies[row] = IsopiesticExperiment(iso.loc[row]).get_experiment(
            reference=reference
        )
    except:
        print("Error on row index {}!".format(row))

#%% Extract osmotic coefficients
molality_ref = []
osmotic_model = []
osmotic_offset = []
salts = []
for cell, ie in ies.items():
    molality_ref += list(np.full(ie.cups, ie.salt_totals[ie.cup_ref].item()))
    osmotic_model += list(ie.osmotic_coefficients["model"])
    osmotic_offset += list(ie.osmotic_coefficients["offset"])
    salts += list(ie.salt_names.values())
osmo = pd.DataFrame(
    {
        "molality_ref": molality_ref,
        "osmotic_model": osmotic_model,
        "osmotic_offset": osmotic_offset,
        "salts": salts,
    }
)
osmo = osmo[osmo.salts != reference]

# Visualise
osmo[(osmo.molality_ref < 5) & (osmo.salts == "KCl")].plot.scatter(
    "molality_ref", "osmotic_offset"
)
