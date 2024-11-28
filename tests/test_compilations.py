import itertools
import jax
from jax import numpy as np
from datetime import datetime as dt
from pytzer import parameters as p, debyehueckel, unsymmetrical
from pytzer.constants import b
from pytzer.convert import solute_to_charge
import pytzer as pz


solutes = pz.model.library.get_solutes()
solutes.update(
    {
        "Na": 0.5,
        "K": 0.5,
        "Cl": 0.5,
        "Ca": 0.5,
        "Mg": 0.5,
        "OH": 0.5,
        "SO4": 0.5,
        "CO3": 0.5,
        "CaF": 0.5,
        "CO2": 0.5,
        "BOH3": 0.5,
        "BOH4": 0.5,
    }
)
solutes = {k: 0.5 for k in solutes}

temperature, pressure = 298.15, 10.1325
gibbs = pz.model.Gibbs_nRT(solutes, temperature, pressure)
# gibbs_old = pz.model.Gibbs_nRT(solutes, temperature, pressure)
print(gibbs)
# print(gibbs_old)

# %%
# solutes = {
#     "Na": 1.1,
#     "Cl": 1.2,
#     "OH": 1.3,
#     "SO4": 1.4,
#     "Br": 1.5,
#     "Mg": 1.6,
#     "Ca": 1.7,
#     "K": 1.8,
#     "Sr": 1.9,
# }
# charges = {
#     "Na": 1,
#     "Cl": -1,
#     "OH": -1,
#     "SO4": -2,
#     "Br": -1,
#     "Mg": 2,
#     "Ca": 2,
#     "K": 1,
#     "Sr": 2,
# }
# solute_to_code = {k: i for i, k in enumerate(solutes)}
# code_to_solute = {i: k for k, i in solute_to_code.items()}
# temperature = 25.0


# def loop_1(solutes, temperature):
#     cations = [s for s in solutes if charges[s] > 0]
#     anions = [s for s in solutes if charges[s] < 1]
#     gibbs = 0.0
#     for cation in cations:
#         for anion in anions:
#             try:
#                 gibbs = gibbs + solutes[cation] * solutes[anion] * library.ca[cation][
#                     anion
#                 ](temperature)
#             except KeyError:
#                 pass
#     return gibbs


# def loop_2(solutes, temperature):
#     gibbs = 0.0
#     for cation in library.ca:
#         for anion in library.ca[cation]:
#             gibbs = gibbs + solutes[cation] * solutes[anion] * library.ca[cation][
#                 anion
#             ](temperature)
#     return gibbs


# def map_1(solutes, temperature):
#     cations = [s for s in solutes if charges[s] > 0]
#     anions = [s for s in solutes if charges[s] < 1]
#     gibbs = 0.0
#     ca = itertools.product(cations, anions)
#     gibbs = gibbs + sum(
#         map(
#             lambda ca: solutes[ca[0]]
#             * solutes[ca[1]]
#             * library.ca[ca[0]][ca[1]](temperature),
#             ca,
#         )
#     )
#     return gibbs


# testscan = np.arange(10)


# def scan_1(solutes, temperature):

#     def scan_func(carry, x):
#         y = testscan[x]
#         return carry + y, y

#     x = np.arange(10)
#     gibbs = jax.lax.scan(scan_func, 0.0, x)
#     return gibbs


# def map_2(solutes, temperature):
#     solutes_coded = np.array(
#         list(
#             map(
#                 lambda k: solutes[k],
#                 solutes.keys(),
#             )
#         )
#     )
#     c_codes = [solute_to_code[s] for s in solutes if charges[s] > 0]
#     a_codes = [solute_to_code[s] for s in solutes if charges[s] < 1]
#     gibbs = 0.0
#     ca = np.array(list(itertools.product(c_codes, a_codes)))
#     gibbs = gibbs + np.sum(
#         jax.lax.map(
#             lambda ca: solutes_coded[ca[0]]
#             * solutes_coded[ca[1]]
#             * library.ca_coded[ca[0]][ca[1]](temperature),
#             ca,
#         )
#     )
#     return gibbs


# #
# gibbs_loop_1 = loop_1(solutes, temperature)
# gibbs_loop_2 = loop_2(solutes, temperature)
# gibbs_map_1 = map_1(solutes, temperature)
# gibbs_scan_1 = scan_1(solutes, temperature)
# # gibbs_map_2 = map_2(solutes, temperature)
# print("loop_1", gibbs_loop_1)
# print("loop_2", gibbs_loop_2)
# print(" map_1", gibbs_map_1)
# # print("map_2 ", gibbs_map_2)

# # %% Test compilation times
# start = dt.now()
# gibbs_loop_1 = jax.jit(loop_1)(solutes, temperature)
# print("Compile loop_1", dt.now() - start)
# start = dt.now()
# gibbs_loop_2 = jax.jit(loop_2)(solutes, temperature)
# print("Compile loop_2", dt.now() - start)
# start = dt.now()
# gibbs_map_1 = jax.jit(map_1)(solutes, temperature)
# print("Compile map_1", dt.now() - start)

# # Test grad times
# start = dt.now()
# grad_loop_1 = jax.grad(loop_1)(solutes, temperature)
# print("Grad loop_1", dt.now() - start)
# start = dt.now()
# grad_loop_2 = jax.grad(loop_2)(solutes, temperature)
# print("Grad loop_2", dt.now() - start)
# start = dt.now()
# grad_map_1 = jax.grad(map_1)(solutes, temperature)
# print("Grad map_1", dt.now() - start)

# # Test compile grad times
# start = dt.now()
# cgrad_loop_1 = jax.jit(jax.grad(loop_1))(solutes, temperature)
# print("Compile grad loop_1", dt.now() - start)
# start = dt.now()
# cgrad_loop_2 = jax.jit(jax.grad(loop_2))(solutes, temperature)
# print("Compile grad loop_2", dt.now() - start)
# start = dt.now()
# cgrad_map_1 = jax.jit(jax.grad(map_1))(solutes, temperature)
# print("Compile grad map_1", dt.now() - start)
