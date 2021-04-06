# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019-2020  Matthew Paul Humphreys  (GNU GPLv3)
import setuptools
from pytzer import __version__

with open("README.md", "r") as f:
    long_description = f.read()
with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()
requirements = [
    "autograd @ {}".format(r) if r.startswith("git") and "autograd" in r else r
    for r in requirements
]
setuptools.setup(
    name="Pytzer",
    version=__version__,
    author="Matthew P. Humphreys",
    author_email="m.p.humphreys@icloud.com",
    description="Pitzer model for chemical activities in aqueous solutions",
    url="https://github.com/mvdh7/pytzer",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Natural Language :: English",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
)
