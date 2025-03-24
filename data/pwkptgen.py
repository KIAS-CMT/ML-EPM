#!/usr/bin/env python3

kfile = open("../KPOINTS", "r")

kpoints =  kfile.readlines()[-1]

with open('KPOINTS.pw', 'w') as f:
    f.write("K_POINTS automatic\n")
    f.write(kpoints.strip())
    f.write(" 0 0 0\n")

