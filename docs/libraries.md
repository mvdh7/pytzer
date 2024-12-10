# Model parameters

Functions that can calculate the parameters required to run a Pitzer model from temperature and pressure are compiled into parameter `Library`s.

## Parameter libraries

### Select library

Select the `Library` that you wish to use with (e.g.):

```python
from pytzer.libraries import Seawater

# Switch to the HWT22 library
pz.set_library(pz, "HWT22")
```

The options are

  * `CWTD23`: [CWTD23](../refs/#c) (**default**)
  * `CHW22`: [CHW22](../refs/#c)
  * `CRP94`: [CRP94](../refs/#c)
  * `HWT22`: [HWT22](../refs/#h)
  * `M88`: [M88](../refs/#m)

Switching to a different parameter library means that all the model functions will be recompiled the next time they are run.
