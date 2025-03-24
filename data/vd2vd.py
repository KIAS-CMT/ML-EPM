#!/usr/bin/env python3

import numpy as np
import os
os.environ["OMP_NUM_THREADS"] = "4"
import subprocess


def cleanq(q):
    tq = np.zeros((len(q),))

    tq[1:] = 1 / q[1:] ** 2
    tq = tq.round(decimals=3)

    _, ind = np.unique(tq, return_index=True)

    ind = np.sort(ind)

    return _, ind


def lv():
    vdata = np.load(f"vdata.npy")

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

    np.save(f"lv.npy", tdata)


def v_s(RCUT):

    vdata = np.load(f"vdata.npy")

    q = vdata[:, 0]
    Nmax = np.where(q > RCUT)[0][0]
    vdata = vdata[1:Nmax, 1:]

    np.save(f"v_s_{RCUT:3.1f}.npy", vdata)


def v_l(RCUT):

    vdata = np.load(f"lv.npy")

    q = vdata[:, 0]

    Nmin = np.where(q > RCUT)[0][0]

    _, ind = cleanq(q)

    ind = ind[ind >= Nmin]

    np.save(f"v_l.npy", vdata[ind])


def v_0():

    vdata = np.load(f"lv.npy")

    vdata = vdata[0, 1:].reshape(1, -1)

    np.save(f"v_0.npy", vdata)


aSi = 13.3394523451361
aO = 12.5492912510208

vdata = np.loadtxt(f"vdata.txt")

np.save(f"vdata.npy", vdata)

lv()
v_s(3.0)
v_l(2.9)
v_0()
