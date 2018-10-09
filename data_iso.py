from autograd import numpy as np
import pytzer as pz
import pandas as pd

isobase = pz.data.iso('datasets/')

isopair = ['KCl','NaCl']

isobase,mols0,mols1,ions0,ions1,T = pz.data.get_isopair(isobase,isopair)
