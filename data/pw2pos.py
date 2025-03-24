#!/usr/bin/env python3

import numpy as np

fname = "pw.scf.in"
# fname = "pw.bands.in"

with open(fname, "r") as f:
    cont = [line.strip().split() for line in f]

cell = []
pos = []

iscar = False

for i, v in enumerate(cont):
    if v[0] == 'nat':
        nat = int(v[2])

    if v[0] == "CELL_PARAMETERS":
        for j in range(i + 1, i + 4):
            t = [float(x) for x in cont[j]]
            cell.append(t)

    if v[0] == "ATOMIC_POSITIONS":
        if v[1][0].lower() == "a":
            iscar = True
        elif v[1][0].lower() == "c":
            iscar = False

        for j in range(i + 1, i + 1 + nat):
            t = [float(x) for x in cont[j][1:]]
            pos.append(t)

cell = np.array(cell)
pos = np.array(pos)

if not iscar:
    tcell = np.transpose(cell)
    tpos = np.zeros(pos.shape)

    for i, _ in enumerate(pos):
        tpos[i] = tcell @ pos[i]

    pos = tpos

# for c in cell:
#     print(c)

# for p in pos:
#     print(p)

nSi = nat//3
nO = nat - nSi

with open("POSCAR", "w") as f:
    f.write(f"SiO2\n")
    f.write(f"1.0\n")
    for c in cell:
        f.write(f"{c[0]:22.16f} {c[1]:22.16f} {c[2]:22.16f}\n")
    f.write(f"Si O\n")
    f.write(f"{nSi} {nO}\n")
    f.write(f"Cartesian\n")
    for c in pos:
        f.write(f"{c[0]:22.16f} {c[1]:22.16f} {c[2]:22.16f}\n")

