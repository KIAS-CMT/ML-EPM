#!/usr/bin/env python3

import os
from dirlist import dirlist

for dname in dirlist:
    index = dname.split("-")[1].split("_")[0]
    # os.system(f"cp run.sh {dname}/ &&"
    #           f"cd {dname} &&"
    #           f"poscarDiCa.py POSCAR &&"
    #           f"sed -i 's|#PBS -N test|#PBS -N {index}|' run.sh &&"
    #           f"qsub run.sh")

    # os.system(f"cp run.pr.sh {dname}/ &&"
    #           f"cd {dname} &&"
    #           f"sed -i 's|#PBS -N test|#PBS -N {index}|' run.pr.sh &&"
    #           f"qsub run.pr.sh")

    # os.system(f"cp run.sh {dname}/ &&"
    #           f"cd {dname} &&"
    #           f"sed -i 's|#PBS -N test|#PBS -N {index}|' run.sh &&"
    #           f"qsub run.sh")


    os.system(f"cp run.sh {dname}/")
    for i in range(8):
        os.system(f"cd {dname} &&"
                  f"cp run.sh run.{i:02d}.sh &&"
                  f"sed -i 's|#PBS -N test|#PBS -N {index}-{i:02d}|' run.{i:02d}.sh &&"
                  f"sed -i 's|dirname|{i:02d}|' run.{i:02d}.sh &&"
                  f"qsub run.{i:02d}.sh")

    # print(dname, index)
