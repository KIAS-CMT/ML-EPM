#!/usr/bin/env python3

import os

_, dirlist, _ = next(os.walk("./"))

dirlist = [dname for dname in dirlist if dname.startswith("mp-")]
#dirlist = [dname for dname in dirlist if dname.startswith("mp-10")]

dirlist.sort()

with open(f"nat.txt", "w") as g:
    for dname in dirlist:
        with open(f"{dname}/POSCAR", "r") as f:
            for _ in range(6):
                f.readline()
            cont = [int(x) for x in f.readline().strip().split()]
            nat = sum(cont)

            g.write(f"{dname} {nat}\n")

