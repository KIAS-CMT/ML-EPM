#!/usr/bin/env python3

import numpy as np
import ase.io
import sys
sys.path.append('/scratch/smkang/dscribe')
#import rdscribe.descriptors
import dscribe.descriptors

SOAP = dscribe.descriptors.SOAP

struct = ase.io.read(f"./POSCAR", format="vasp")
struct.wrap(eps=1e-8)

desc1 = SOAP(
    species=["Si", "O"],
    rcut=10.0,
    nmax=7,
    lmax=7,
    periodic=True,
    crossover=True,
    sparse=False,
    coeff=True,
)

# desc2 = SOAP(
#     species=["Si", "O"],
#     rcut=5.0,
#     nmax=6,
#     lmax=6,
#     periodic=True,
#     crossover=True,
#     sparse=False,
# )

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

dc = desc1.create(struct)
soap = desc.create(struct)

np.save(f"./dc.npy", dc)
np.save(f"./soap.npy", soap)
