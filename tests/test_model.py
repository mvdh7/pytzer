from collections import OrderedDict
import pytzer as pz
from pytzer.libraries import Seawater

# Define test conditions
solutes = OrderedDict(
    {
        "Na": 1.0,
        "Cl": 1.0,
        "Ca": 0.5,
        "SO4": 1.0,
        "Mg": 0.5,
        "tris": 1.0,
    }
)  # molalities in mol/kg
temperature = 300  # K
pressure = 100  # dbar

# Get pz.model function arguments
kwargs = dict(temperature=temperature, pressure=pressure, verbose=False)
params = Seawater.get_parameters(solutes, **kwargs)


def test_parameter_library():
    """Does the parameter library have the expected type and methods?"""
    assert isinstance(Seawater, pz.ParameterLibrary)
    assert isinstance(params, dict)
    for v in ["temperature", "pressure", "Aphi", "ca"]:
        assert v in params


def test_model_funcs():
    """Do all the main model functions return floats?"""
    Gibbs_nRT = pz.model.Gibbs_nRT(solutes, **params)
    assert isinstance(Gibbs_nRT.item(), float)
    log_activity_water = pz.model.log_activity_water(solutes, **params)
    assert isinstance(log_activity_water.item(), float)
    activity_water = pz.model.activity_water(solutes, **params)
    assert isinstance(activity_water.item(), float)
    osmotic_coefficient = pz.model.osmotic_coefficient(solutes, **params)
    assert isinstance(osmotic_coefficient.item(), float)
    log_activity_coefficients = pz.model.log_activity_coefficients(solutes, **params)
    activity_coefficients = pz.model.activity_coefficients(solutes, **params)
    for s in solutes:
        assert isinstance(log_activity_coefficients[s].item(), float)
        assert isinstance(activity_coefficients[s].item(), float)


def test_toplevel_funcs():
    """Do all the top-level functions return floats?"""
    Gibbs_nRT = pz.Gibbs_nRT(solutes, **params)
    assert isinstance(Gibbs_nRT.item(), float)
    log_activity_water = pz.log_activity_water(solutes, **params)
    assert isinstance(log_activity_water.item(), float)
    activity_water = pz.activity_water(solutes, **params)
    assert isinstance(activity_water.item(), float)
    osmotic_coefficient = pz.osmotic_coefficient(solutes, **params)
    assert isinstance(osmotic_coefficient.item(), float)
    log_activity_coefficients = pz.log_activity_coefficients(solutes, **params)
    activity_coefficients = pz.activity_coefficients(solutes, **params)
    for s in solutes:
        assert isinstance(log_activity_coefficients[s].item(), float)
        assert isinstance(activity_coefficients[s].item(), float)


# test_parameter_library()
# test_model_funcs()
# test_toplevel_funcs()
