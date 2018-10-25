import setuptools

with open('README.md','r') as fh:
    long_description = fh.read()

setuptools.setup(
    name         = 'pytzer',
    version      = '0.1.11',
    author       = 'Matthew P. Humphreys',
    author_email = 'm.p.humphreys@cantab.net',
    description  = 'Pitzer model for chemical speciation in natural waters',
    url          = 'https://github.com/mvdh7/pytzer',
    packages     = setuptools.find_packages(),
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    classifiers = (
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',),)
