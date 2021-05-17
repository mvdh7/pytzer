# Pytzer v0.4.3

[![pypi badge](https://img.shields.io/pypi/v/pytzer.svg?style=popout)](https://pypi.org/project/pytzer/)

***May 2021: a totally new version is in development!  If you want to use Pytzer, I advise you to wait for that.***

Pitzer model for chemical activities in aqueous solutions. Undergoing beta testing and development.

**Installation:**

    pip install pytzer
    pip install git+https://github.com/mvdh7/autograd#egg=autograd --upgrade --no-cache-dir

The second line above is strongly recommended, but optional. It upgrades [Autograd](https://github.com/HIPS/autograd) to the latest version that has been tested with Pytzer, which eliminates some deprecation warnings that may appear when using the relatively old Autograd version available from PyPI. You could also switch `mvdh7` in the URL to `HIPS` to get the very latest Autograd straight from the horse's mouth.

**Documentation:** [pytzer.readthedocs.io](https://pytzer.readthedocs.io/en/latest/), including a [quick-start guide](https://pytzer.readthedocs.io/en/latest/quick-start/).

Pytzer is implemented and maintained by [Matthew P. Humphreys](https://mvdh.xyz) at the NIOZ Royal Netherlands Institute for Sea Research (Texel, the Netherlands).

**Citation:**

For currently released v0.4.3:

> Humphreys, Matthew P. (2019).  Pytzer: the Pitzer model for chemical activities in aqueous solutions in Python (beta).  Version 0.4.3.  *Zenodo.* [doi:10.5281/zenodo.3404907](https://doi.org/10.5281/zenodo.3404907).
