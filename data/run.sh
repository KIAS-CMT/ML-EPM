#!/bin/bash

#PBS -N test
#PBS -l walltime=10:00:00
#PBS -l nodes=1:ppn=24
#PBS -q normal
#PBS -o stdout.log

cd $PBS_O_WORKDIR
module load intel/19.1.3.304
val=`cat ntyp.txt`

nstr=$((val-1))

# nstr1=$((4*val))
# nstr2=$((6*val-1))

# for i in {00..07}
for i in dirname

do

for j in 00 $(seq -f "%02g" 1 $((nstr)))
# for j in $(seq -f "%02g" $((nstr1)) $((nstr2)))

do
mkdir ${i}-dis-${j}
cd ${i}-dis-${j}
mkdir ./tmp
../../pwkptgen.py
../../pwposgen.dis.py ${i}
../../pwingen.py ${i}-dis-${j}
mpirun -machinefile $PBS_NODEFILE /home/smkang/qe-6.8/bin/pw.x -nk 2 < pw.scf.in > pw.scf.out
/scratch/smkang/code.pp/bin/pp.x < pp.in
python ../../vd2vd.py
python ../../pw2pos.py
python ../../cal_feat.py
rm vdata.txt
rm -rf ./tmp
cd ..
done
done
