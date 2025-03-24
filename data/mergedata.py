#!/usr/bin/env python3

import os
os.environ["OMP_NUM_THREADS"] = "6"

import subprocess
import numpy as np
import concurrent.futures
from sklearn.utils import shuffle
import joblib
from dirlist import dirlist


def load_v_s_ind(root, dname, vol, j):

    index = dname.split("-")[1].split("_")[0]

    if j == "pr":
        fdname = f"{dname}/{vol}-pr/"
    else:
        fdname = f"{dname}/{vol}-dis-{j}/"

    print(f"Processing {fdname}...")

    vdata = np.load(f"{root}/{fdname}/v_s_{gcut:3.1f}.npy")

    nq = len(vdata)

    ntyp = (len(vdata[0]) - 5) // 2

    g = vdata[:, :3]
    s = vdata[:, 3:-2]
    v = vdata[:, -2:]

    coeff = np.zeros((1, ntyp))

    for i in range(ntyp):
        key = f"{index}-{vol}-{j}-{i:02d}"
        coeff[0, i] = desc[key]

    coeff = np.tile(coeff, (nq, 1))

    if ntyp != N:
        nd = N - ntyp
        coeff = np.hstack((coeff, np.zeros((nq, nd))))
        s = np.hstack((s, np.zeros((nq, 2 * nd))))

    data = np.hstack((coeff, g, s, v))

    return data


def load_v_s(root, dname, vol, j):

    if j == "pr":
        fdname = f"{dname}/{vol}-pr/"
    else:
        fdname = f"{dname}/{vol}-dis-{j}/"

    print(f"Processing {fdname}...")

    coeff = np.load(f"{root}/{fdname}/dc.npy")

    ntyp = len(coeff)
    lens = len(coeff[0])

    vdata = np.load(f"{root}/{fdname}/v_s.npy")

    nq = len(vdata)

    g = vdata[:, :3]
    s = vdata[:, 3:-2]
    v = vdata[:, -2:]

    coeff = np.reshape(coeff, (1, -1))
    coeff = np.tile(coeff, (nq, 1))

    if ntyp != N:
        nd = N - ntyp
        coeff = np.hstack((coeff, np.zeros((nq, nd * lens))))
        s = np.hstack((s, np.zeros((nq, 2 * nd))))

    data = np.hstack((coeff, g, s, v))

    return data


def load_v_sph_ind(root, dname, vol, j):

    index = dname.split("-")[1].split("_")[0]

    if j == "pr":
        fdname = f"{dname}/{vol}-pr/"
    else:
        fdname = f"{dname}/{vol}-dis-{j}/"

    print(f"Processing {fdname}...")

    vdata = np.load(f"{root}/{fdname}/v_sph.npy")

    nq = len(vdata)

    ntyp = len(vdata[0]) - 2

    g = vdata[:, 0].reshape(-1, 1)
    s = vdata[:, 1:-1]
    v = vdata[:, -1].reshape(-1, 1)

    soap = np.zeros((1, ntyp))

    for i in range(ntyp):
        key = f"{index}-{vol}-{j}-{i:02d}"
        soap[0, i] = desc[key]

    soap = np.tile(soap, (nq, 1))

    if ntyp != N:
        nd = N - ntyp
        soap = np.hstack((soap, np.zeros((nq, nd))))
        s = np.hstack((s, np.zeros((nq, nd))))

    data = np.hstack((soap, g, s, v))

    return data


def load_v_l_ind(root, dname, vol, j):

    index = dname.split("-")[1].split("_")[0]

    if j == "pr":
        fdname = f"{dname}/{vol}-pr/"
    else:
        fdname = f"{dname}/{vol}-dis-{j}/"

    print(f"Processing {fdname}...")

    vdata = np.load(f"{root}/{fdname}/v_l.npy")

    nq = len(vdata)

    ntyp = len(vdata[0]) - 2

    g = vdata[:, 0].reshape(-1, 1)
    s = vdata[:, 1:-1]
    v = vdata[:, -1].reshape(-1, 1)

    soap = np.zeros((1, ntyp))

    for i in range(ntyp):
        key = f"{index}-{vol}-{j}-{i:02d}"
        soap[0, i] = desc[key]

    soap = np.tile(soap, (nq, 1))

    if ntyp != N:
        nd = N - ntyp
        soap = np.hstack((soap, np.zeros((nq, nd))))
        s = np.hstack((s, np.zeros((nq, nd))))

    data = np.hstack((soap, g, s, v))

    return data


def load_v_l(root, dname, vol, j):

    if j == "pr":
        fdname = f"{dname}/{vol}-pr/"
    else:
        fdname = f"{dname}/{vol}-dis-{j}/"

    print(f"Processing {fdname}...")

    soap = np.load(f"{root}/{fdname}/soap.npy")

    ntyp = len(soap)
    lens = len(soap[0])

    vdata = np.load(f"{root}/{fdname}/v_l.npy")

    nq = len(vdata)

    g = vdata[:, 0].reshape(-1, 1)
    s = vdata[:, 1:-1]
    v = vdata[:, -1].reshape(-1, 1)

    soap = np.reshape(soap, (1, -1))
    soap = np.tile(soap, (nq, 1))

    if ntyp != N:
        nd = N - ntyp
        soap = np.hstack((soap, np.zeros((nq, nd * lens))))
        s = np.hstack((s, np.zeros((nq, nd))))

    data = np.hstack((soap, g, s, v))

    return data


def load_v_0(root, dname, vol, j):

    if j == "pr":
        fdname = f"{dname}/{vol}-pr/"
    else:
        fdname = f"{dname}/{vol}-dis-{j}/"

    print(f"Processing {fdname}...")

    soap = np.load(f"{root}/{fdname}/soap.npy")

    ntyp = len(soap)
    lens = len(soap[0])

    vdata = np.load(f"{root}/{fdname}/v_0.npy")

    nq = 1

    s = vdata[:, :-1]
    v = vdata[:, -1].reshape(-1, 1)

    soap = np.reshape(soap, (1, -1))
    soap = np.tile(soap, (nq, 1))

    if ntyp != N:
        # m, n = np.divmod(N, ntyp)
        # soap = np.tile(soap, (1, m))
        # soap = np.hstack((soap, np.zeros((nq, n * lens))))
        # s = np.tile(s, (1, m)) / m
        # s = np.hstack((s, np.zeros((nq, n))))

        nd = N - ntyp
        soap = np.hstack((soap, np.zeros((nq, nd * lens))))
        s = np.hstack((s, np.zeros((nq, nd))))

    data = np.hstack((soap, s, v))

    return data


def main1():
    args = []
    for dname in dirlist:
        nat = np.loadtxt(f"{dname}/ntyp.txt", dtype=int)
        nvol = 8
        nstr = nat
        for i in [f"{t:02d}" for t in range(nvol)]:
            for j in [f"{t:02d}" for t in range(nstr)]:
                args.append(("./", dname, i, j))

    data = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for d in executor.map(wrapper, args):
            data.append(d)

    data = np.vstack(data)

    print(data.shape)

    data = shuffle(data)

    np.save(f"{savename}.npy", data)


def main2():
    for key, dname in dic.items():
        nat = np.loadtxt(f"{dname}/ntyp.txt", dtype=int)
        nvol = 8
        nstr = nat
        for i in [f"{t:02d}" for t in range(nvol)]:
            args = []
            for j in [f"{t:02d}" for t in range(nstr)]:
                args.append(("./", dname, i, j))

            savename = f"{key:02d}-{i}"

            data = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
                for d in executor.map(wrapper, args):
                    data.append(d)

            data = np.vstack(data)

            data = shuffle(data)

            np.save(f"./data/raw/{savename}.npy", data)


# dic = {}
# for i, v in enumerate(dirlist):
#     dic[i] = v

desc = joblib.load("descriptor.dic")

if __name__ == "__main__":

    gcut = 3.0

    N = 24

    savename = "d_l"

    def wrapper(arg):
        # ret = load_v_sph_ind(*arg)
        # ret = load_v_s(*arg)
        # ret = load_v_l(*arg)
        # ret = load_v_0(*arg)
        # ret = load_v_s_ind(*arg)
        ret = load_v_l_ind(*arg)
        return ret

    main1()
    # main2()
