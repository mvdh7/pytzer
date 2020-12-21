import pytzer as pz
from pytzer.libraries import Seawater

# Define test conditions
solute_molalities = {
    "Na": 1.0,
    "Cl": 1.0,
    "Ca": 0.5,
    "SO4": 1.0,
    "Mg": 0.5,
    "tris": 1.0,
}  # molalities in mol/kg
temperature = 300  # K
pressure = 100  # dbar

# Get pz.model function arguments
args, ss = pz.get_pytzer_args(solute_molalities)
kwargs = dict(temperature=temperature, pressure=pressure, verbose=False)
params = Seawater.get_parameters(**ss, **kwargs)


def test_model_functions():
    """Do all the model functions return floats?"""
    Gibbs_nRT = pz.model.Gibbs_nRT(*args, **params)
    assert isinstance(Gibbs_nRT.item(), float)
    log_activity_water = pz.model.log_activity_water(*args, **params)
    assert isinstance(log_activity_water.item(), float)
    activity_water = pz.model.activity_water(*args, **params)
    assert isinstance(activity_water.item(), float)
    osmotic_coefficient = pz.model.osmotic_coefficient(*args, **params)
    assert isinstance(osmotic_coefficient.item(), float)
    log_activity_coefficients = pz.model.log_activity_coefficients(*args, **params)
    for x in log_activity_coefficients:
        assert isinstance(x[0].item(), float)
    activity_coefficients = pz.model.activity_coefficients(*args, **params)
    for x in activity_coefficients:
        assert isinstance(x[0].item(), float)


def test_wrap_functions():
    """Do all the wrap functions return floats?"""
    Gibbs_nRT = pz.Gibbs_nRT(solute_molalities, **kwargs)
    assert isinstance(Gibbs_nRT.item(), float)
    log_activity_water = pz.log_activity_water(solute_molalities, **kwargs)
    assert isinstance(log_activity_water.item(), float)
    activity_water = pz.activity_water(solute_molalities, **kwargs)
    assert isinstance(activity_water.item(), float)
    osmotic_coefficient = pz.osmotic_coefficient(solute_molalities, **kwargs)
    assert isinstance(osmotic_coefficient.item(), float)
    log_activity_coefficients = pz.log_activity_coefficients(
        solute_molalities, **kwargs
    )
    for x in log_activity_coefficients:
        assert isinstance(x[0].item(), float)
    activity_coefficients = pz.activity_coefficients(solute_molalities, **kwargs)
    for x in activity_coefficients:
        assert isinstance(x[0].item(), float)


# test_model_functions()
# test_wrap_functions()
