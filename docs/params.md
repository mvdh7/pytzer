# Model parameters

Functions that can calculate the parameters required to run a Pitzer model from temperature and pressure are compiled into `ParameterLibrary`s, which can be imported from `pytzer.libraries`.

## Parameter libraries

### Import library

Import the `ParameterLibrary` that you wish to use with (e.g.):

```python
from pytzer.libraries import Seawater
```

The options are:

  * `Clegg23`: [CHW23](../refs/#c) (**default**)
  * `Clegg22`: [CHW22](../refs/#c)
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

### Update parameter library

Changing or updating the parameter library has to be done using `pz.update_library`:

```python
import pytzer as pz
from pytzer.libraries import Seawater

# Changes can be made to the Seawater library here if needed

pz.update_library(pz, Seawater)
```

This step can be a bit slow, as all the model functions must be recompiled afterwards.  It only needs to be repeated if the parameter library being used is changed.
