## Installation with Anaconda/Miniconda

[1] Download and install [Anaconda](https://www.anaconda.com/distribution/) (or [Miniconda](https://conda.io/en/latest/miniconda.html) if you prefer). Use the Python 3.X version if you have no preference, but either will do.

[2] Open an Anaconda prompt window (Windows) or the Terminal (Mac/Linux).

[3] Create a new environment for Pytzer:

    conda create -n pytzer python=3.7.3 numpy=1.16.1 scipy=1.2.1

Other recent versions of Python, NumPy and SciPy will probably work fine too, but they have not been tested with Pytzer.

[<span id="inst4">4</span>] Activate the new environment:

    conda activate pytzer    # Windows or Linux
    source activate pytzer   # Mac

You should now see the name of the environment (i.e. `pytzer`) appear at the start of each new line in the command window.

[5] Install Pytzer into the `pytzer` environment:

    pip install pytzer
    pip install git+https://github.com/mvdh7/autograd#egg=autograd --upgrade --no-cache-dir

The second line above is strongly recommended, but optional. It upgrades [Autograd](https://github.com/HIPS/autograd) to the latest version that has been tested with Pytzer, which eliminates some deprecation warnings that may appear when using the relatively old Autograd version available from PyPI. You could also switch `mvdh7` in the URL to `HIPS` to get the very latest Autograd straight from the horse's mouth.

<hr />

## Running Pytzer as a "black box"

### Without equilibration

You can just provide Pytzer with a CSV file of temperature, pressures and molality values, and have it return the corresponding activity coefficients in a new CSV file, only having to write the bare minimum in Python, as follows.

[1] Create an input CSV file as described for `getmols` in the [import/export documentation](../modules/io/#getmols-import-csv-dataset) - or just save and use the example file [pytzerQuickStart.csv](https://raw.githubusercontent.com/mvdh7/pytzer/master/testfiles/pytzerQuickStart.csv).

[2] Open an Anaconda prompt window (Windows) or the Terminal (Mac/Linux) and activate the `pytzer` environment (i.e. [installation](#installation-with-anacondaminiconda) step [4]).

[3] Navigate to the folder containing the input CSV file ([using cd](https://en.wikipedia.org/wiki/Cd_(command))).

[4] Start Python:

    python

[5] Import Pytzer (as `pz` by convention) and then run its "black-box" function on your input file:

```python
>>> import pytzer as pz
>>> pz.blackbox('pytzerQuickStart.csv')
```

Once the calculations are complete, a new CSV file will appear, in the same folder as the input file, with the same name as the input file but with `_py` appended. It contains the input temperature and molality values, followed by the osmotic coefficient (column header: `osm`), water activity (`aw`), and then the activity coefficient of each solute (with column headers e.g. `gNa` for the $\gamma(\ce{Na+})$ activity coefficient).

The black box function is currently set up to use the Seawater [parameter library](../modules/libraries). This is based on the MIAMI model of Millero and Pierrot [[MP98](../refs/#m)], with a few modifications. It is still under development, so the results will probably change as Pytzer is updated. The Seawater coefficient library contains coefficients for the components: $\ce{Na+}$, $\ce{K+}$, $\ce{Ca^2+}$, $\ce{Mg^2+}$, $\ce{MgOH+}$, $\ce{H+}$, $\ce{OH-}$, $\ce{Cl-}$, $\ce{HSO4-}$, $\ce{SO4^2-}$, $\ce{tris}$, and $\ce{trisH+}$. If you have any other solutes in your input file, Pytzer should still work (as long the charges on those solutes are defined in [properties.charges](../modules/properties/#charges-solute-charges)), but all interaction coefficients involving those solutes will be set to zero.

### With equilibration

There is a separate black-box function that works in a similar way, but also includes solving for equilibrium:

```python
>>> pz.blackbox_equilibrate('trisASWequilibrium.csv')
```

This function requires the input file to be formatted as described for the `gettots` import function ([see documentation here](../modules/io/#gettots-import-csv-dataset)).

<hr />

## Updating Pytzer

[1] Open an Anaconda prompt window (Windows) or the Terminal (Mac/Linux) and activate the pytzer environment ([installation step [4]](#inst4)).

[2] Update Pytzer with pip:

    pip install pytzer --upgrade --no-cache-dir
