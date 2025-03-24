#!/usr/bin/env python3

import os
import re
import sys
import numpy as np

# scf = open("scf.in", 'r').read()
pos = open("POSCAR.pw", 'r').read()
kpt = open("KPOINTS.pw", 'r').read()

nat = len(pos.split("\n")) - 6
prefix = os.getcwd().split("/")[-2].split("-")[-1].split("_")[0] + '-' +sys.argv[-1]

# re1 = re.compile(r"(prefix) = '(.*)'")
# re2 = re.compile(r"(nat) = (\d+)")
# scf = re1.sub(f"prefix = '{prefix}'", scf)
# scf = re2.sub(f"nat = {nat}", scf)

Si_ind = np.loadtxt("../Si_ind.txt", unpack=True)
O_ind = np.loadtxt("../O_ind.txt", unpack=True)
#ntyp = np.loadtxt("../ntyp.txt", unpack=True)
ntyp = np.loadtxt("../ntyp.txt", unpack=True, dtype=int)

with open("pw.scf.in", 'w') as f:
    f.write(f'''&control
prefix = '{prefix}',
calculation = 'scf'
restart_mode='from_scratch',
! verbosity = 'high'
pseudo_dir = '/scratch/smkang/PP/'
outdir = './tmp/'
/
&system
ibrav = 0
! nbnd = 24
ntyp = {ntyp}
nat =  {nat}
ecutwfc = 84
/
&electrons
diagonalization = 'david'
mixing_mode = 'plain'
mixing_beta = 0.7
conv_thr =  1.0d-8
/
ATOMIC_SPECIES
''')

    Si_u = len(np.unique(Si_ind))
    O_u = len(np.unique(O_ind))
    for i in range(Si_u):
        f.write(f"Si{i:.0f} 1 Si.pbe-n-rrkjus_psl.1.0.0.UPF\n")
    for i in range(O_u):
        f.write(f"O{i:.0f} 1 O.pbe-n-kjpaw_psl.0.1.UPF\n")

    f.write(pos)
    f.write(kpt)

with open('pp.in', 'w') as f:
    f.write(f'''&inputpp
prefix='{prefix}',
outdir = './tmp/'
plot_num = 1
/
''')

os.system('rm POSCAR.pw')
os.system('rm KPOINTS.pw')
