import itertools
from jax import numpy as np, vmap, jit, lax
from time import time

cats = np.vstack([1.0, 2, 3, 4, 5])
anis = np.array([[5.0, 2, 6]])
ca = np.array([[3.0, 1, 4, 7, 5], [3, 2, 6, 8, 4], [1, 4, 3, 6, 8]])

m_cats = cats.ravel()
m_anis = anis.ravel()

go = time()


@jit
def test_matmul():
    anisanis = np.array(
        [(np.transpose(anis) @ anis)[np.triu_indices(np.size(anis), k=1)]]
    )
    # anix = np.triu_indices(np.size(anis), k=1)
    # anisanis = anis[0][anix[0]] * anis[0][anix[1]]
    catscats = np.array(
        [(np.transpose(cats) @ cats)[np.triu_indices(np.size(cats), k=1)]]
    )
    return (anis @ ca @ cats)[0][0]


test_matmul()
print(time() - go)
go = time()
test_matmul()
print(time() - go)


def map_func(i_ca):
    ic, ia = i_ca
    return m_cats[ic] * m_anis[ia] * ca[ia][ic]


go = time()


@jit
def test_map():
    r_cats = range(len(m_cats))
    r_anis = range(len(m_anis))
    i_ca = np.array(list(itertools.product(r_cats, r_anis)))
    i_cc = np.array(list(itertools.product(r_cats, r_cats)))
    i_aa = np.array(list(itertools.product(r_anis, r_anis)))
    return np.sum(lax.map(map_func, i_ca))


test_map()
print(time() - go)
go = time()
test_map()
print(time() - go)
