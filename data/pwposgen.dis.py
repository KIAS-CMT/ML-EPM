#!/usr/bin/env python3

import sys
import numpy as np
import itertools

pos = open('../POSCAR', 'r')
# pos = open('%s' % sys.argv[1], 'r')

pos.readline()
pos.readline()

sindex = sys.argv[-1]

a = [0.975, 1.025]

sdic = {}
for u, v in enumerate(itertools.product(*itertools.repeat(a, 3))):
    sdic[f"{u:02d}"] = v

s1, s2, s3 = sdic[sindex]

veca = np.array([float(x) * s1 for x in pos.readline().split()])
vecb = np.array([float(x) * s2 for x in pos.readline().split()])
vecc = np.array([float(x) * s3 for x in pos.readline().split()])

mat = np.zeros((3,3))
mat[:,0] = veca
mat[:,1] = vecb
mat[:,2] = vecc

name = pos.readline().split()
atoms = [int(x) for x in pos.readline().split()]
ntype = len(name)

nat = 0
for n in atoms:
    nat = nat + n

# print(" ntyp = %s" % ntype)
# print(" nat = %s" % nat)

pos.readline()

outfile = open('POSCAR.pw', 'w')
# outfile = open('%s.pw' % sys.argv[1], 'w')
# outfile = open('%s.%s.%s.pw' % (sys.argv[1], ntype, nat), 'w')

# outfile.write("ntyp = %s\n" % ntype)
# outfile.write("nat = %s\n" % nat)

# tobohr 1.889725989

outfile.write('CELL_PARAMETERS angstrom\n')
outfile.write('%22.16f%22.16f%22.16f\n' % (veca[0], veca[1], veca[2]))
outfile.write('%22.16f%22.16f%22.16f\n' % (vecb[0], vecb[1], vecb[2]))
outfile.write('%22.16f%22.16f%22.16f\n' % (vecc[0], vecc[1], vecc[2]))

outfile.write("ATOMIC_POSITIONS angstrom\n")


def random_vector():
    # r = 0.00
    r = 0.1

    theta = np.random.uniform(0, np.pi)
    phi = np.random.uniform(0, 2 * np.pi)

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])


Si_ind = np.loadtxt("../Si_ind.txt", unpack=True)
O_ind = np.loadtxt("../O_ind.txt", unpack=True)

for v in Si_ind:
    temp = np.array([float(x) for x in pos.readline().split()[0:3]])
    # temp = mat @ temp
    temp = temp + random_vector()
    outfile.write("%s %22.16f%22.16f%22.16f\n" % ("Si" + str(int(v)), temp[0], temp[1], temp[2]))

for v in O_ind:
    temp = np.array([float(x) for x in pos.readline().split()[0:3]])
    # temp = mat @ temp
    temp = temp + random_vector()
    outfile.write("%s %22.16f%22.16f%22.16f\n" % ("O" + str(int(v)), temp[0], temp[1], temp[2]))

pos.close()
outfile.close()
