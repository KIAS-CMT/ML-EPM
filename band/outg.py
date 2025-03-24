#!/usr/bin/env python3

import numpy as np

gcut = 3.0

gvec = np.loadtxt("gvec.txt")
gl = np.sqrt(np.sum(gvec**2, axis=1))
Nmax = np.where(gl > gcut)[0][0]
gvec = gvec[1:Nmax]

with open("gvec_t.txt", "w") as f:
    for val in gvec:
        f.write(f"{val[0]:15.8E} {val[1]:15.8E} {val[2]:15.8E}\n")

with open("ngvec_t.txt", "w") as f:
    val = len(gvec)
    f.write(f"{val}\n")


g = np.unique(np.around(gl, decimals=7))
Nmax = np.where(g > gcut)[0][0]
g = np.hstack((g[0], g[Nmax:]))

with open("g_t.txt", "w") as f:
    for val in g:
        f.write(f"{val:15.8E}\n")

with open("ng_t.txt", "w") as f:
    val = len(g)
    f.write(f"{val}\n")
