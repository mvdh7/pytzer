import numpy as np
import pytzer as pz

cf = pz.cfdicts.WM13_MarChemSpec25

# Compare beta-0 values
b0file = 'print_coeffs/GIVAKT-beta0.txt'
b0ions = np.genfromtxt(b0file,'U', usecols=[0,1])
b0vals = np.genfromtxt(b0file, usecols=2)

T = 298.15

b0pz = np.array([cf.bC['-'.join(ionpair)](T)[0] for ionpair in b0ions])

b0diffs = b0vals - b0pz

print('===== beta-0 =====')
print('{:^8} {:^13} {:^13} {:^13}'.format('ions','GIVAKT','pytzer','diff'))

for i, ionpair in enumerate(b0ions):
    
    ix = ''
    if np.abs(b0diffs[i]) > 0:
        ix = '*'
    
    print('{:>8} {:>13.6e} {:>13.6e} {:>13.6e} {}'.format('-'.join(ionpair),
                                   b0vals[i],
                                   b0pz[i],
                                   b0diffs[i],
                                   ix))

print('')

# Compare beta-1 values
b1file = 'print_coeffs/GIVAKT-beta1.txt'
b1ions = np.genfromtxt(b1file,'U', usecols=[0,1])
b1vals = np.genfromtxt(b1file, usecols=2)
b1alph1 = np.genfromtxt(b1file, usecols=3)

T = 298.15

b1pz = np.array([cf.bC['-'.join(ionpair)](T)[1] for ionpair in b1ions])

b1diffs = b1vals - b1pz

print('===== beta-1 =====')
print('{:^8} {:^13} {:^13} {:^13}'.format('ions','GIVAKT','pytzer','diff'))

for i, ionpair in enumerate(b1ions):
    
    ix = ''
    if np.abs(b1diffs[i]) > 0:
        ix = '*'
    
    print('{:>8} {:>13.6e} {:>13.6e} {:>13.6e} {}'.format('-'.join(ionpair),
                                   b1vals[i],
                                   b1pz[i],
                                   b1diffs[i],
                                   ix))

print('')

# Compare beta-2 values
b2file = 'print_coeffs/GIVAKT-beta2.txt'
b2ions = np.genfromtxt(b2file,'U', usecols=[0,1])
b2vals = np.genfromtxt(b2file, usecols=2)
b2alph2 = np.genfromtxt(b2file, usecols=3)

T = 298.15

b2pz = np.array([cf.bC['-'.join(ionpair)](T)[2] for ionpair in b2ions])

b2diffs = b2vals - b2pz

print('===== beta-2 =====')
print('{:^8} {:^13} {:^13} {:^13}'.format('ions','GIVAKT','pytzer','diff'))

for i, ionpair in enumerate(b2ions):
    
    ix = ''
    if np.abs(b2diffs[i]) > 0:
        ix = '*'
    
    print('{:>8} {:>13.6e} {:>13.6e} {:>13.6e} {}'.format('-'.join(ionpair),
                                   b2vals[i],
                                   b2pz[i],
                                   b2diffs[i],
                                   ix))

print('')

# Compare C-phi values
Cphifile = 'print_coeffs/GIVAKT-Cphi.txt'
Cphiions = np.genfromtxt(Cphifile,'U', usecols=[0,1])
Cphivals = np.genfromtxt(Cphifile, usecols=2)

T = 298.15

C0pz = np.array([cf.bC['-'.join(ionpair)](T)[3] for ionpair in Cphiions])

zs = np.concatenate((np.vstack(pz.props.charges(Cphiions[:,0])[0]),
                     np.vstack(pz.props.charges(Cphiions[:,1])[0])),
                    axis=1)

Cphipz = np.array([C0 * 2 * np.sqrt(np.prod(np.abs(zs[i]))) \
                   for i,C0 in enumerate(C0pz)])

Cphidiffs = Cphivals - Cphipz

print('===== C-phi =====')
print('{:^8} {:^13} {:^13} {:^13}'.format('ions','GIVAKT','pytzer','diff'))

for i, ionpair in enumerate(Cphiions):
    
    ix = ''
    if np.abs(Cphidiffs[i]) > 0:
        ix = '*'
    
    print('{:>8} {:>13.6e} {:>13.6e} {:>13.6e} {}'.format('-'.join(ionpair),
                                   Cphivals[i],
                                   Cphipz[i],
                                   Cphidiffs[i],
                                   ix))

print('')

# Compare C-0 values
C0file = 'print_coeffs/GIVAKT-C0.txt'
C0ions = np.genfromtxt(C0file,'U', usecols=[0,1])
C0vals = np.genfromtxt(C0file, usecols=2)

T = 298.15

C0pz = np.array([cf.bC['-'.join(ionpair)](T)[3] for ionpair in C0ions])

C0diffs = C0vals - C0pz

print('===== C-0 =====')
print('{:^8} {:^13} {:^13} {:^13}'.format('ions','GIVAKT','pytzer','diff'))

for i, ionpair in enumerate(C0ions):
    
    ix = ''
    if np.abs(C0diffs[i]) > 0:
        ix = '*'
    
    print('{:>8} {:>13.6e} {:>13.6e} {:>13.6e} {}'.format('-'.join(ionpair),
                                   C0vals[i],
                                   C0pz[i],
                                   C0diffs[i],
                                   ix))

print('')

# Compare C-1 values
C1file = 'print_coeffs/GIVAKT-C1.txt'
C1ions = np.genfromtxt(C1file,'U', usecols=[0,1])
C1vals = np.genfromtxt(C1file, usecols=2)

T = 298.15

C1pz = np.array([cf.bC['-'.join(ionpair)](T)[4] for ionpair in C1ions])

C1diffs = C1vals - C1pz

print('===== C-1 =====')
print('{:^8} {:^13} {:^13} {:^13}'.format('ions','GIVAKT','pytzer','diff'))

for i, ionpair in enumerate(C1ions):
    
    ix = ''
    if np.abs(C1diffs[i]) > 0:
        ix = '*'
    
    print('{:>8} {:>13.6e} {:>13.6e} {:>13.6e} {}'.format('-'.join(ionpair),
                                   C1vals[i],
                                   C1pz[i],
                                   C1diffs[i],
                                   ix))

print('')

# Compare theta values
thetafile = 'print_coeffs/GIVAKT-theta.txt'
thetaions = np.genfromtxt(thetafile,'U', usecols=[0,1])
thetavals = np.genfromtxt(thetafile, usecols=2)

T = 298.15

for i, ionpair in enumerate(thetaions):
    
    ionpair.sort()
    
    thetaions[i] = ionpair

thetapz = np.array([cf.theta['-'.join(ionpair)](T)[0] for ionpair in thetaions])

thetadiffs = thetavals - thetapz

print('===== theta =====')
print('{:^8} {:^13} {:^13} {:^13}'.format('ions','GIVAKT','pytzer','diff'))

for i, ionpair in enumerate(thetaions):
    
    ix = ''
    if np.abs(thetadiffs[i]) > 0:
        ix = '*'
    
    print('{:>8} {:>13.6e} {:>13.6e} {:>13.6e} {}'.format('-'.join(ionpair),
                                   thetavals[i],
                                   thetapz[i],
                                   thetadiffs[i],
                                   ix))

print('')
