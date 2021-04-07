import itertools, jax
from jax import numpy as np, lax
import numpy as onp


def raiser(x, y, xy):
    return (x + y) * xy


# @jax.jit
def get_tot_loop(xs, ys, mx):
    nx, ny = np.shape(mx)
    tot = 0.0
    for ix, iy in itertools.product(range(nx), range(ny)):
        tot = tot + raiser(xs[ix], ys[iy], mx[ix, iy])
    return tot


@jax.jit
def get_tot_map(xs, ys, mx):
    nx, ny = np.shape(mx)

    def raiser_ix(ix_iy):
        ix, iy = ix_iy
        return raiser(xs[ix], ys[iy], mx[ix, iy])

    tot = 0.0
    ix_iy = np.array(list(itertools.product(range(nx), range(ny))))
    tot = tot + np.sum(lax.map(raiser_ix, ix_iy))
    tot = tot + np.sum(lax.map(raiser_ix, ix_iy))

    return tot


# def get_tot_scan(xs, ys, mx):


nx, ny = 50, 60
xs = np.array(onp.random.normal(size=nx))
ys = np.array(onp.random.normal(size=ny))
mx = np.array(onp.random.normal(size=(nx, ny)))

# print("loop 1")
# tot_loop = get_tot_loop(xs, ys, mx).item()
# print("loop 2")
# tot_loop = get_tot_loop(xs, ys, mx).item()
print("map 1")
tot_map = get_tot_map(xs, ys, mx).item()
print("map 2")
tot_map = get_tot_map(xs, ys, mx).item()
print("grad map")
grad_map = jax.jit(jax.grad(get_tot_map))(xs, ys, mx)
