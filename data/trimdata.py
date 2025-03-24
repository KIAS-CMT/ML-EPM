#!/usr/bin/env python3

import os
os.environ["OMP_NUM_THREADS"] = "4"

import subprocess
import numpy as np
import concurrent.futures
import itertools
from sklearn.utils import shuffle
from dirlist import dirlist


def cleanq(q):
    tq = np.zeros((len(q),))

    tq[1:] = 1 / q[1:] ** 2
    tq = tq.round(decimals=3)

    _, ind = np.unique(tq, return_index=True)

    ind = np.sort(ind)

    return _, ind


def lv(root, dname, vol, j):

    # index = dname.split("-")[1].split("_")[0]

    if j == "pr":
        fdname = f"{dname}/{vol}-pr/"
    else:
        fdname = f"{dname}/{vol}-dis-{j}/"

    print(f"Processing {fdname}...")

    vdata = np.load(f"{root}/{fdname}/vdata.npy")

    nat = (vdata.shape[1] - 6) // 2

    q = np.around(vdata[:, 0], decimals=7)
    # lenq = len(q)

    assert np.all(q[:-1] <= q[1:])

    q_u, ind, count = np.unique(q, return_index=True, return_counts=True)
    lenq_u = len(q_u)

    assert np.isclose(q_u[0], 0.0)
    assert count[0] == 1

    tdata = np.zeros((lenq_u, nat + 2))

    tdata[:, 0] = q_u

    for i in range(lenq_u):
        tdata[i, 1:] = np.mean(vdata[ind[i]:ind[i] + count[i], 4:-1:2], axis=0)

    np.save(f"{root}/{fdname}/lv.npy", tdata)


def v_s(root, dname, vol, j):

    # index = dname.split("-")[1].split("_")[0]

    if j == "pr":
        fdname = f"{dname}/{vol}-pr/"
    else:
        fdname = f"{dname}/{vol}-dis-{j}/"

    print(f"Processing {fdname}...")

    vdata = np.load(f"{root}/{fdname}/vdata.npy")

    q = vdata[:, 0]
    Nmax = np.where(q > RCUT)[0][0]
    vdata = vdata[1:Nmax, 1:]

    np.save(f"{root}/{fdname}/v_s_{RCUT:3.1f}.npy", vdata)


def v_l(root, dname, vol, j):

    # index = dname.split("-")[1].split("_")[0]

    if j == "pr":
        fdname = f"{dname}/{vol}-pr/"
    else:
        fdname = f"{dname}/{vol}-dis-{j}/"

    print(f"Processing {fdname}...")

    vdata = np.load(f"{root}/{fdname}/lv.npy")

    q = vdata[:, 0]

    Nmin = np.where(q > RCUT)[0][0]

    _, ind = cleanq(q)

    ind = ind[ind >= Nmin]

    np.save(f"{root}/{fdname}/v_l_{RCUT:3.1f}.npy", vdata[ind])


def v_0(root, dname, vol, j):

    # index = dname.split("-")[1].split("_")[0]

    if j == "pr":
        fdname = f"{dname}/{vol}-pr/"
    else:
        fdname = f"{dname}/{vol}-dis-{j}/"

    print(f"Processing {fdname}...")

    vdata = np.load(f"{root}/{fdname}/lv.npy")

    # omega = float(
    #     subprocess.getoutput(
    #         f"grep 'unit-cell volume' {root}/{fdname}/pw.scf.out"
    #     ).split()[3]
    # )

    # ntyp = len(vdata[0]) - 2

    # nSi = ntyp // 3
    # nO = ntyp - nSi

    vdata = vdata[0, 1:].reshape(1, -1)

    # vdata[0, -1] = vdata[0, -1] - (nSi * aSi + nO * aO) / omega

    np.save(f"{root}/{fdname}/v_0_woal.npy", vdata)


def v_sph(root, dname, vol, j):

    # index = dname.split("-")[1].split("_")[0]

    if j == "pr":
        fdname = f"{dname}/{vol}-pr/"
    else:
        fdname = f"{dname}/{vol}-dis-{j}/"

    print(f"Processing {fdname}...")

    vdata = np.load(f"{root}/{fdname}/lv.npy")

    q = vdata[:, 0]

    _, ind = cleanq(q)

    # omega = float(
    #     subprocess.getoutput(
    #         f"grep 'unit-cell volume' {root}/{fdname}/pw.scf.out"
    #     ).split()[3]
    # )

    # ntyp = len(vdata[0]) - 2

    # nSi = ntyp // 3
    # nO = ntyp - nSi

    # vdata[0, -1] = vdata[0, -1] - (nSi * aSi + nO * aO) / omega

    np.save(f"{root}/{fdname}/v_sph.npy", vdata[ind])


if __name__ == "__main__":

    RCUT = 3.0

    aSi = 13.3394523451361
    aO = 12.5492912510208

    args = []
    for dname in dirlist:
        nat = np.loadtxt(f"{dname}/ntyp.txt", dtype=int)
        nvol = 8
        nstr = nat
        for i in [f"{t:02d}" for t in range(nvol)]:
            for j in [f"{t:02d}" for t in range(nstr)]:
                args.append(("./", dname, i, j))

    def wrapper(arg):
        v_0(*arg)
        # v_s(*arg)
        # v_l(*arg)
        # v_sph(*arg)

    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        executor.map(wrapper, args)
