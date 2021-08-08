from itertools import product
import pytzer as pz, numpy as np
import jax

xs = [0.5, 1.0, 2.5, 10.0]
funcs = [pz.unsymmetrical.P75_eq46, pz.unsymmetrical.P75_eq47, pz.unsymmetrical.Harvie]


def test_grads():
    """Are all the functions grad-able?"""
    for x, func in product(xs, funcs):
        assert isinstance(jax.grad(func)(x).item(), float)


def test_jit_grads():
    """Are all the grad-ed functions jit-able?"""
    for x, func in product(xs, funcs):
        assert isinstance(jax.jit(jax.grad(func))(x).item(), float)


def test_Harvie_values():
    """Do Harvie and grad(Harvie) return the exact correct values compared
    with the _Harvie_raw function?
    """
    for x in xs:
        j__Harvie_raw, jg__Harvie_raw = [
            j.item() for j in pz.unsymmetrical._Harvie_raw(x)
        ]
        j__Harvie = pz.unsymmetrical.Harvie(x)
        jg__Harvie = jax.grad(pz.unsymmetrical.Harvie)(x)
        assert j__Harvie_raw == j__Harvie
        assert jg__Harvie_raw == jg__Harvie


def test_values():
    """Do all functions return similar values (with 1% tolerance)?"""
    for x in xs:
        js = np.array([func(x).item() for func in funcs])
        js_mean = np.mean(js)
        assert np.allclose(js, js_mean, atol=0, rtol=0.01)


def test_grad_values():
    """Do all grad-ed functions return similar values (with 2% tolerance)?"""
    for x in xs:
        jgs = np.array([jax.grad(func)(x).item() for func in funcs])
        jgs_mean = np.mean(jgs)
        assert np.allclose(jgs, jgs_mean, atol=0, rtol=0.02)


def test_Harvie_checks():
    """Do Harvie and grad(Harvie) return the check values given by P91 p. 125?"""
    j = pz.unsymmetrical.Harvie(1.0)
    assert np.round(j, decimals=6) == 0.116437
    jg = jax.grad(pz.unsymmetrical.Harvie)(1.0)
    assert np.round(jg, decimals=6) == 0.160527


# test_grads()
# test_jit_grads()
# test_Harvie_values()
# test_values()
# test_grad_values()
# test_Harvie_checks()
