#!/usr/bin/env python3

import os
import numpy as np
import ase.io
import ase.spacegroup
import ase.spacegroup.symmetrize
from dirlist import dirlist

def ftn(dname, mode):
    struct = ase.io.read(f"{dname}/POSCAR", format="vasp")
    nat = len(struct.get_positions())

    if mode == 1:
        sym = ase.spacegroup.symmetrize.check_symmetry(struct, symprec=1e-4)
        tmp = sym['equivalent_atoms']
        Si_ = tmp[:nat // 3]
        O_ = tmp[nat // 3:]
        Si_u = np.unique(Si_)
        O_u = np.unique(O_)
        Si_ind = np.zeros(len(Si_), dtype=int)
        O_ind = np.zeros(len(O_), dtype=int)

        for i, v in enumerate(Si_u):
            Si_ind[Si_ - v == 0] = i

        for i, v in enumerate(O_u):
            O_ind[O_ - v == 0] = i

        ntyp = len(Si_u) + len(O_u)

    elif mode == 2:
        nSi = nat // 3
        nO = nat - nSi

        Si_ind = list(range(nSi))
        O_ind = list(range(nO))

        ntyp = nat

    with open(f"{dname}/ntyp.txt", "w") as f:
        f.write(f"{ntyp}\n")

    with open(f"{dname}/Si_ind.txt", "w") as f:
        for v in Si_ind:
            f.write(f"{v}\n")

    with open(f"{dname}/O_ind.txt", "w") as f:
        for v in O_ind:
            f.write(f"{v}\n")

mode = 2
for dname in dirlist:
    ftn(dname, mode)
