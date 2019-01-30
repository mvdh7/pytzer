import setuptools
from pytzer import __version__

with open('README.md','r') as fh:
    long_description = fh.read()

setuptools.setup(
    name         = 'pytzer',
    version      = __version__,
    author       = 'Matthew P. Humphreys',
    author_email = 'm.p.humphreys@cantab.net',
    description  = 'Pitzer model for chemical activities in aqueous solutions',
    url          = 'https://github.com/mvdh7/pytzer',
    packages     = ['pytzer',
                    'pytzer.cfdicts',
                    'pytzer.coeffs',
                    'pytzer.constants',
                    'pytzer.io',
                    'pytzer.jfuncs',
                    'pytzer.meta',
                    'pytzer.model',
                    'pytzer.props'],
    install_requires = ['autograd==1.2',
                        'numpy>=1.15'],
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    classifiers = (
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 3.6',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
        'Natural Language :: English',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Chemistry'),)
