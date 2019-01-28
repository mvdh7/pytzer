from autograd import grad

def y(x): return 3*x**2 + 4.5*x + 2
dy_dx   = grad(y)
d2y_dx2 = grad(dy_dx)

test = d2y_dx2(3.)

y_0       = y      (0.)
dy_dx_0   = dy_dx  (0.)
d2y_dx2_0 = d2y_dx2(0.)

mc = d2y_dx2_0 / 2
mb = dy_dx_0 + 0. * d2y_dx2_0

my = lambda x: y_0 + mb*x + mc*x**2

tval = 34.
test = [y(tval), my(tval)]
