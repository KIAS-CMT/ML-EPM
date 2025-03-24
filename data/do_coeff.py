#!/usr/bin/env python3

import os
os.environ["OMP_NUM_THREADS"] = "4"

import concurrent.futures
import subprocess
import numpy as np
import ase.io
import ase.spacegroup
import ase.spacegroup.symmetrize
import rdscribe.descriptors
import dscribe.descriptors
from dirlist import dirlist

# from rascal.representations import SphericalCovariants, SphericalInvariants
# from ase import Atoms
# from ase.build import molecule
# from ase.spacegroup import get_spacegroup

SOAP = dscribe.descriptors.SOAP


def ftn(root, dname, i, j):
    if j == "pr":
        fdname = f"{root}/{dname}/{i}-pr/"
    else:
        fdname = f"{root}/{dname}/{i}-dis-{j}/"

    print(f"Processing {fdname}...")

    os.system(f"cd {fdname} &&"
              f"../../pw2pos.py")

    struct = ase.io.read(f"{fdname}/POSCAR", format="vasp")

    struct.wrap(eps=1e-8)

    desc = SOAP(
        species=["Si", "O"],
        rcut=10.0,
        nmax=7,
        lmax=7,
        periodic=True,
        crossover=True,
        sparse=False,
        coeff=True,
    )
    dc = desc.create(struct)
    np.save(f"{fdname}/dc.npy", dc)

    desc = SOAP(
        species=["Si", "O"],
        # rcut=5.0,
        nmax=6,
        lmax=6,
        weighting={"function": "poly", "c": 1.0, "m": 2.0, "r0": 10.0},
        periodic=True,
        crossover=True,
        sparse=False,
    )
    soap = desc.create(struct)
    np.save(f"{fdname}/soap.npy", soap)


if __name__ == "__main__":
    args = []
    for dname in dirlist:
        nat = np.loadtxt(f"{dname}/ntyp.txt", dtype=int)
        nvol = 8
        nstr = nat
        for i in [f"{t:02d}" for t in range(nvol)]:
            for j in [f"{t:02d}" for t in range(nstr)]:
                args.append(("./", dname, i, j))

    def wrapper(arg):
        ftn(*arg)

    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        executor.map(wrapper, args)
