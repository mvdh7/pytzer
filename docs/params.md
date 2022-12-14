# Model parameters

Functions that can calculate the parameters required to run a Pitzer model from temperature and pressure are compiled into `ParameterLibrary`s, which can be imported from `pytzer.libraries`.

## Parameter libraries

### Import library

Import the `ParameterLibrary` that you wish to use with (e.g.):

```python
from pytzer.libraries import Seawater
```

The options are:

  * `Clegg94`: [CRP94](../refs/#c)
  * `Greenberg89`: [GM89](../refs/#m)
  * `Harvie84`: [HMW84](../refs/#h)
  * `Humphreys22`: [HWT22](../refs/#h)
  * `MarChemSpec`
  * `MarChemSpec25`
  * `Millero98`: [MP98](../refs/#m), a.k.a. MIAMI
  * `Moller88`: [M88](../refs/#m)
  * `Seawater`
  * `Waters13`: [WM13](../refs/#w)
  * `Waters13_MarChemSpec25`

### Update unsymmetrical mixing function

Different `ParameterLibrary`s are designed for use with different *J* functions for unsymmetrical mixing.  To change which function is being used throughout Pytzer, use (e.g.):

```python
import pytzer as pz
from pytzer.libraries import Seawater

pz = Seawater.set_func_J(pz)
```

This step can be slow, as all the model functions must be recompiled afterwards.  It only needs to be repeated if the `ParameterLibrary` being used is changed.

### Evaluate parameters

To generate the `params` dict required for various functions in Pytzer, use (e.g.):

```python
from pytzer.libraries import Seawater

params = Seawater.get_parameters(
    solutes=None, temperature=298.15, pressure=10.1023, verbose=True
)
```

The arguments are:

  * `solutes`: an `OrderedDict` where the keys represent the solutes in the solution, as described in the [model arguments](../model/#arguments).  If `None`, then parameters are evaluated for all solutes represented by the `ParameterLibrary`'s functions.
  * `temperature`: in K.
  * `pressure`: in dbar.
  * `verbose`: do you want to print a warning to stdout whenever there is no function in the `ParameterLibrary` for any interaction between the `solutes`?

If you are running Pytzer over multiple different solutions, then you need to re-run `get_parameters()` between each solution only if the temperature, pressure, or combination of solutes has changed.  In other words, if only the molality values for the solutes have changed between calculations, then you do not need to re-run `get_parameters()`.
