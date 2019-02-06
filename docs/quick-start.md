# Installation with Anaconda/Miniconda

[1] Download and install [Anaconda](https://www.anaconda.com/distribution/) (or [Miniconda](https://conda.io/en/latest/miniconda.html) if you prefer). Use the Python 3.X version if you have no preference, but either will do.

[2] Open an Anaconda prompt window (Windows) or the Terminal (Mac/Linux).

[3] Create a new environment for **pytzer** by entering the following:

```
conda create -n pytzer python=3.6 numpy
```

[4] Activate the new environment. On Windows:

```
activate pytzer
```

On Mac/Linux:

```
source activate pytzer
```

You should now see the name of the environment (i.e. `pytzer`) appear at the very start of each new line in the command window.

[5] Install **pytzer** into the new environment:

```
pip install pytzer
```

<hr />

# Running pytzer as a "black box"

You can just provide **pytzer** with a CSV file of temperatures and molality values, and have it return the corresponding activity coefficients in a new CSV file, only having to write the bare minimum in Python, as follows.

[1] Create an input CSV file as described in the [import/export documentation](../modules/io/#pytzeriogetmols) - or just save and use the example file [pytzerQuickStart.csv](https://raw.githubusercontent.com/mvdh7/pytzer/master/testfiles/pytzerQuickStart.csv).

[2] Open an Anaconda prompt window (Windows) or the Terminal (Mac/Linux) and activate the pytzer environment ([installation](#installation-with-anacondaminiconda) step [4]).

[3] Navigate to the folder containing the input CSV file ([using cd](https://en.wikipedia.org/wiki/Cd_(command))).

[4] Start Python:

```
python
```

[5] Import **pytzer**:

```python
import pytzer as pz
```

[6] Run **pytzer** on your input file:

```python
pz.blackbox('pytzerQuickStart.csv')
```

Once the calculations are complete, a new CSV file will appear, in the same folder as the input file, with the same name as the input file but with **\_out** appended. It contains the input temperature and molality values, followed by the osmotic coefficient (column header: **osm**), water activity (**aw**), and then the activity coefficient of each solute (with column headers e.g. **gNa** for Na<sup>+</sup> activity coefficient).

The black box function is currently set up to use the **MarChemSpec** [coefficient dictionary](../modules/cfdicts). This is based on the model of Waters and Millero (2013), with some additional terms added for tris interactions. It is still under development, so the results <u>will</u> change as **pytzer** is updated. It contains coefficients for the components: Ca<sup>2+</sup>, Cl<sup>−</sup>, H<sup>+</sup>, HSO<sub>4</sub><sup>−</sup>, K<sup>+</sup>, Mg<sup>2+</sup>, MgOH<sup>+</sup>, Na<sup>+</sup>, OH<sup>−</sup>, SO<sub>4</sub><sup>−</sup>, tris, and trisH<sup>+</sup>. If you have any ions in your input file that are not on this list, then **pytzer** should still work (as long the charges on those ions are defined in **props.charges**), but all interaction coefficients involving those ions will be set to zero.

[7] To exit Python, and return to the Anaconda prompt or Terminal:

```python
exit()
```

<hr />

# Updating pytzer

[1] Open an Anaconda prompt window (Windows) or the Terminal (Mac/Linux) and activate the pytzer environment ([installation](#installation-with-anacondaminiconda) step [4]).

[2] Update **pytzer** with pip:

```
pip install pytzer --upgrade --no-cache-dir
```
