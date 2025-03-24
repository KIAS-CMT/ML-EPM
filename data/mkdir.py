#!/usr/bin/env python3

import os
import sys
import numpy as np


def read_str():
    with open("./skel/nat.txt", "r") as f:
        cont = [line.strip().split() for line in f]

    cont = [[u, int(v)] for u, v in cont]

    return cont


str = read_str()

dic = {
    6: [
        [4, 1, 1],
        [1, 4, 1],
        [1, 1, 4],
        [2, 2, 1],
        [2, 1, 2],
        [1, 2, 2],
    ],
    9: [
        [2, 1, 1],
        [1, 2, 1],
        [1, 1, 2],
    ],
    12: [
        [2, 1, 1],
        [1, 2, 1],
        [1, 1, 2],
    ],
    24: [
        [1, 1, 1],
    ]
}


def read_kpts(dname):
    with open(f"{dname}/KPOINTS", "r") as f:
        cont = [line.strip() for line in f]

    p, q, r = [float(x) for x in cont[-1].split()]

    return cont[-2], (p, q, r)


for dname, nat in str:
    index = dname[3:].split('_')[0]
    for l in dic[nat]:
        nx, ny, nz = l
        nd = f"mp-{index}.{nx}x{ny}x{nz}_SiO2"
        os.system(f"mkdir {nd}")
        os.system(f"cp ./skel/{dname}/POSCAR {nd}/")
        os.system(f"cd {nd};"
                  f"python ../poscarsupercell.py POSCAR {nx} {ny} {nz};"
                  f"mv POSCAR{nx}{ny}{nz} POSCAR")
        p, q = read_kpts(f"./skel/{dname}/")
        q = [int(np.ceil(u / v)) for u, v in zip(q, l)]
        with open(f"{nd}/KPOINTS", "w") as f:
            f.write("K\n")
            f.write("0\n")
            f.write(f"{p}\n")
            f.write(f"{q[0]} {q[1]} {q[2]}\n")
