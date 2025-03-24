#!/usr/bin/env python3

import os
os.environ["OMP_NUM_THREADS"] = "4"
import concurrent.futures
import itertools
import subprocess
import joblib
import numpy as np
from sklearn.utils import shuffle
from dirlist import dirlist


def gendesdic():
    def ftn(dname, vol, j, nat):
        nonlocal giter

        index = dname.split("-")[1].split("_")[0]

        for i in range(nat):
            key = f"{index}-{vol}-{j}-{i:02d}"
            dic[key] = giter
            giter = giter + 1

#    giter = 232416
    giter = 0

    dic = {}

    for dname in dirlist:
        nat = np.loadtxt(f"{dname}/ntyp.txt", dtype=int)
        nvol = 8
        nstr = nat
        for i in [f"{t:02d}" for t in range(nvol)]:
            for j in [f"{t:02d}" for t in range(nstr)]:
                ftn(dname, i, j, nat)

    print(len(dic))

    joblib.dump(dic, "descriptor.dic")


def gendesarr(desc):
    def load(root, dname, vol, j):

        if j == "pr":
            fdname = f"{dname}/{vol}-pr/"
        else:
            fdname = f"{dname}/{vol}-dis-{j}/"

        des = np.load(f"{root}/{fdname}/{desc}.npy")

        return des

    data = []

    for dname in dirlist:
        nat = np.loadtxt(f"{dname}/ntyp.txt", dtype=int)
        nvol = 8
        nstr = nat
        for i in [f"{t:02d}" for t in range(nvol)]:
            print(f"Processing {dname}-{i}")
            for j in [f"{t:02d}" for t in range(nstr)]:
                data.append(load("./", dname, i, j))

    data = np.vstack(data)

    np.save(f"{desc}_all.npy", data)

    print(data.shape)


gendesdic()

gendesarr("soap")

gendesarr("dc")
