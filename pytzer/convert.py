
from jax import numpy as np
from . import constants


def osmotic_to_activity(molalities, osmotic_coefficient):
    """Convert osmotic coefficient to water activity."""
    return np.exp(-osmotic_coefficient * constants.Mw * np.sum(molalities))


def activity_to_osmotic(molalities, activity_water):
    """Convert water activity to osmotic coefficient."""
    return -np.log(activity_water) / (constants.Mw * np.sum(molalities))


def log_activities_to_mean(log_acf_M, log_acf_X, n_M, n_X):
    """Calculate the mean activity coefficient for an electrolyte."""
    return (n_M * log_acf_M + n_X * log_acf_X) / (n_M + n_X)
