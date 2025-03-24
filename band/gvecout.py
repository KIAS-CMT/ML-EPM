#!/usr/bin/env python3

import numpy as np
import sys

try:
    fname = sys.argv[1]
except:
    fname = "pw.bands.in"

with open(fname, "r") as f:
    cont = [line.strip().split() for line in f]

cell = []

for i, v in enumerate(cont):
    if v[0] == 'ecutwfc':
        ecutwfc = float(v[2])

    if v[0] == "CELL_PARAMETERS":
        for j in range(i + 1, i + 4):
            t = [float(x) for x in cont[j]]
            cell.append(t)

atob = 1 / 0.529177210903  # angstrom to bohr

cell = np.array(cell) * atob

inv = 2 * np.pi * np.linalg.inv(cell)

rec = np.transpose(inv)

ecutpot = 4 * ecutwfc  # qe default

b1m, b2m, b3m = (1.5 * np.ceil(np.sqrt(ecutpot / np.sum(rec**2, axis=1)))).astype(int)

arr = [np.arange(-x, x + 1) for x in [b1m, b2m, b3m]]

m = np.stack(np.meshgrid(*arr, indexing="ij"), -1).reshape(-1, 3)

gvec = m @ rec

gg = np.sum(gvec**2, axis=1)

isin = (gg <= ecutpot)

gvec = gvec[isin]
gg = gg[isin]

arg = np.argsort(gg)

gvec = gvec[arg]

np.savetxt("gvec.txt", gvec, fmt='%15.8E')
