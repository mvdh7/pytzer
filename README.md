# Pytzer

[![pypi badge](https://img.shields.io/pypi/v/pytzer.svg?style=popout)](https://pypi.org/project/pytzer/)

Pitzer model for chemical activities in aqueous solutions. Undergoing beta testing and development.

**Installation:**

```shell
pip install pytzer
pip install git+https://github.com/mvdh7/autograd#egg=autograd --upgrade --no-cache-dir
```

The second line above is strongly recommended, but optional. It upgrades [Autograd](https://github.com/HIPS/autograd) to the latest version that has been tested with Pytzer, which eliminates some deprecation warnings that may appear when using the relatively old Autograd version available from PyPI. You could also switch `mvdh7` in the URL to `HIPS` to get the very latest Autograd straight from the horse's mouth.

**Documentation:** [pytzer.readthedocs.io](https://pytzer.readthedocs.io/en/latest/), including a [quick-start guide](https://pytzer.readthedocs.io/en/latest/quick-start/).

Pytzer is implemented and maintained by [Matthew P. Humphreys](https://mvdh.xyz) at the University of East Anglia (Norwich, UK).
