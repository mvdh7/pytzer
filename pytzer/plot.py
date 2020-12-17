# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2020  Matthew Paul Humphreys  (GNU GPLv3)
"""Visualise different parameter sets and calculations."""
from autograd.numpy import array, full, full_like, linspace, nan, sqrt
from autograd.numpy import min as np_min
from autograd.numpy import max as np_max
from . import libraries, meta, model, properties

def _bC_plot(ax, xtype, tots, mols, ele, ions, tempK, pres, prmlib_base,
        varout):
    ifuncs = meta.getifuncs('bC', ions)
    fvar = full((len(ifuncs), len(tempK)), nan)
    fvarfuncs = {
        'acf_anion': (
            lambda mols, ions, tempK, pres, prmlib:
                model.acfs(mols, ions, tempK, pres, prmlib=prmlib)[1],
            '{} activity coefficient in {}'.format(ions[1], ele),
        ),
        'act_anion': (
            lambda mols, ions, tempK, pres, prmlib: mols[1]*
                model.acfs(mols, ions, tempK, pres, prmlib=prmlib)[1],
            '{} activity in {}'.format(ions[1], ele),
        ),
        'acf_cation': (
            lambda mols, ions, tempK, pres, prmlib:
                model.acfs(mols, ions, tempK, pres, prmlib=prmlib)[0],
            '{} activity coefficient in {}'.format(ions[0], ele),
        ),
        'act_cation': (
            lambda mols, ions, tempK, pres, prmlib: mols[0]*
                model.acfs(mols, ions, tempK, pres, prmlib=prmlib)[0],
            '{} activity in {}'.format(ions[0], ele),
        ),
        'aw': (model.aw, 'Water activity in {}'.format(ele)),
        'osm': (model.osm, 'Osmotic coefficient in {}'.format(ele)),
    }
    for i, ifunc in enumerate(ifuncs):
        prmlib_base.bC['-'.join(ions)] = ifuncs[ifunc]
        fvar[i] = fvarfuncs[varout][0](mols, ions, tempK, pres,
            prmlib=prmlib_base)
    xtypes = {
        'tot': (sqrt(tots), 'sqrt(Molality / mol/kg)'),
        'tempK': (tempK, 'Temperature / K'),
        'pres': (pres, 'Pressure / dbar'),
    }
    xvar = xtypes[xtype][0]
    for i, ifunc in enumerate(ifuncs):
        ax.plot(xvar, fvar[i], label=ifunc.split('_')[-1])
        ax.legend()
        ax.set_xlabel(xtypes[xtype][1])
        ax.set_ylabel(fvarfuncs[varout][1])
        ax.set_xlim([np_min(xvar), np_max(xvar)])
    return fvar

def bC_pres(ax, tot, ele, tempK, pres0, pres1,
        prmlib_base=libraries.Seawater, varout='osm'):
    ions, nus = properties._ele2ions[ele]
    pres = linspace(pres0, pres1, 100)
    tots = full_like(pres, tot)
    mols = array([tots*nus[0], tots*nus[1]])
    tempK = full_like(pres, tempK)
    fvar = _bC_plot(ax, 'pres', tots, mols, ele, ions, tempK, pres,
        prmlib_base, varout)
    return fvar

def bC_tempK(ax, tot, ele, tempK0, tempK1, pres,
        prmlib_base=libraries.Seawater, varout='osm'):
    ions, nus = properties._ele2ions[ele]
    tempK = linspace(tempK0, tempK1, 100)
    tots = full_like(tempK, tot)
    mols = array([tots*nus[0], tots*nus[1]])
    pres = full_like(tempK, pres)
    fvar = _bC_plot(ax, 'tempK', tots, mols, ele, ions, tempK, pres,
        prmlib_base, varout)
    return fvar

def bC_tot(ax, tot0, tot1, ele, tempK, pres,
        prmlib_base=libraries.Seawater, varout='osm'):
    ions, nus = properties._ele2ions[ele]
    tots = linspace(sqrt(tot0), sqrt(tot1), 100)**2
    mols = array([tots*nus[0], tots*nus[1]])
    tempK = full_like(tots, tempK)
    pres = full_like(tots, pres)
    fvar = _bC_plot(ax, 'tot', tots, mols, ele, ions, tempK, pres,
        prmlib_base, varout)
    return fvar
