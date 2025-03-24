#!/usr/bin/env python3

import sys

Na = int(sys.argv[2])
Nb = int(sys.argv[3])
Nc = int(sys.argv[4])

pos = open('%s' % sys.argv[1], 'r')

pos.readline()

scale = float(pos.readline().split()[0])

veca = [float(x) * scale for x in pos.readline().split()]
vecb = [float(x) * scale for x in pos.readline().split()]
vecc = [float(x) * scale for x in pos.readline().split()]

name = pos.readline().split()
atoms = [int(x) for x in pos.readline().split()]

Natoms = 0
for n in atoms:
    Natoms += n

supercell = open('%s%d%d%d' % (sys.argv[1], Na, Nb, Nc), 'w')

supercell.write('SYSTEM\n')
supercell.write('1.0\n')
supercell.write('%15.10f%15.10f%15.10f\n' % (Na * veca[0], Na * veca[1], Na * veca[2]))
supercell.write('%15.10f%15.10f%15.10f\n' % (Nb * vecb[0], Nb * vecb[1], Nb * vecb[2]))
supercell.write('%15.10f%15.10f%15.10f\n' % (Nc * vecc[0], Nc * vecc[1], Nc * vecc[2]))
supercell.write(' '.join(name))
supercell.write('\n')

for i in range(len(atoms)):
    if i == len(atoms) - 1:
        supercell.write('%d\n' % (Na * Nb * Nc * atoms[i]))
    else:
        supercell.write('%d ' % (Na * Nb * Nc * atoms[i]))

supercell.write("Cartesian\n")

if pos.readline()[0].lower() == 'd':
    for i in range(len(atoms)):
        for j in range(atoms[i]):
            temp = [float(x) for x in pos.readline().split()[0:3]]
            x = temp[0] * veca[0] + temp[1] * vecb[0] + temp[2] * vecc[0]
            y = temp[0] * veca[1] + temp[1] * vecb[1] + temp[2] * vecc[1]
            z = temp[0] * veca[2] + temp[1] * vecb[2] + temp[2] * vecc[2]

            for n in range(Nc):
                for m in range(Nb):
                    for l in range(Na):
                        xx = x + l * veca[0] + m * vecb[0] + n * vecc[0]
                        yy = y + l * veca[1] + m * vecb[1] + n * vecc[1]
                        zz = z + l * veca[2] + m * vecb[2] + n * vecc[2]
                        supercell.write("%15.10f%15.10f%15.10f\n" % (xx, yy, zz))

else:
    for i in range(len(atoms)):
        for j in range(atoms[i]):
            temp = [float(x) for x in pos.readline().split()[0:3]]
            x = scale * temp[0]
            y = scale * temp[1]
            z = scale * temp[2]

            for n in range(Nc):
                for m in range(Nb):
                    for l in range(Na):
                        xx = x + l * veca[0] + m * vecb[0] + n * vecc[0]
                        yy = y + l * veca[1] + m * vecb[1] + n * vecc[1]
                        zz = z + l * veca[2] + m * vecb[2] + n * vecc[2]
                        supercell.write("%15.10f%15.10f%15.10f\n" % (xx, yy, zz))

pos.close()
supercell.close()

# fname = '%s%d%d%d' % (sys.argv[1], Na, Nb, Nc)

# os.system('poscarDiCa.py %s'%fname)
# os.system('mv %s.conv %s'%(fname,fname))
