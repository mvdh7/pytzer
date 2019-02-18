#from copy import deepcopy
import pytzer as pz
import numpy as np 

# Import test dataset
filepath = 'testfiles/'
filestem = 'pytzerPitzer'

# Run black box calculation (uses MarChemSpec cfdict)
mols,ions,T,cf,Gex_nRT,osm,aw,acfs = pz.blackbox(filepath + filestem + '.csv')

# Load results from Julia
jfilename = filepath + filestem + '_jl.csv'
jdata = np.genfromtxt(jfilename, delimiter=',', skip_header=1)
jhead = np.genfromtxt(jfilename, delimiter=',', skip_footer=np.shape(jdata)[0],
                      dtype='str')
                  
gions = [''.join(('g',ion)) for ion in ions]    
jacfs = np.concatenate([np.vstack(jdata[:,C]) 
    for C in range(np.shape(jdata)[1])
    if jhead[C] in gions],
    axis=1)

# Compare and plot differences
dacfs = jacfs - acfs

from matplotlib import pyplot as plt

fig,ax = plt.subplots()

vminmax = np.max(np.abs(dacfs))
cax = ax.imshow(dacfs, cmap="coolwarm", aspect='auto',
                vmin=-vminmax, vmax=vminmax)
fig.colorbar(cax, label='Diff. in acf')

plt.xticks(range(len(ions)), rotation=0)
ax.set_xticklabels(ions)

ax.set_ylabel('Row in file')
