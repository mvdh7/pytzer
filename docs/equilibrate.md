# Solving equilibria

There are two 'layers' of solver in Pytzer: stoichiometric and thermodynamic.

The stoichiometric solver determines the molality of each solute given a set of total molalities and fixed stoichiometric equilibrium constants.  It uses a Newton-Raphson iterative method that is fully compatible with JAX.

The thermodynamic solver wraps the stoichiometric solver and adjusts the stoichiometric equilibrium constants to agree with thermodynamic constraints.  Because it uses [`scipy.optimize.root`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root.html), it cannot be differentiated or compiled with JAX.
