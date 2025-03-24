#!/usr/bin/env python

import os
import sys
import numpy as np
import tensorflow as tf

os.environ["OMP_NUM_THREADS"] = "12"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import warnings
warnings.filterwarnings("ignore")


def lload():
    dc = np.load(f"./dc.npy")
    soap = np.load(f"./soap.npy")

    nat = len(soap)

    gvec = np.loadtxt("gvec_t.txt")
    gl = np.loadtxt("g_t.txt")

    nq_s = len(gvec)
    nq_l = len(gl) - 1

    v_s = [None] * nat
    v_l = [None] * nat

#    nSi = nat // 3
#    nO = nat - nSi

    for i in range(nat):
        v_s[i] = np.zeros((nq_s,), dtype=complex)
        v_l[i] = np.zeros((nq_l + 1,), dtype=float)

        inp_s = np.hstack((np.tile(dc[i], (nq_s, 1)), gvec[:, :]))

        inp_0 = np.tile(soap[i], (1, 1))
        inp_l = np.hstack((np.tile(soap[i], (nq_l, 1)), gl[1:].reshape(-1, 1)))

        predict_s = model_s.predict(inp_s)
        v_s[i][:] = predict_s[:, 0] + predict_s[:, 1] * 1j

        v_l[i][0] = 0

        predict_l = model_l.predict(inp_l)
        v_l[i][1:] = predict_l[:, 0]

    return nat, v_s, v_l


sname = sys.argv[-1]

model_s = tf.keras.models.load_model(
    f"/scratch/smkang/gst_primitive/ml/v_s/name.save").get_layer("potential")
model_l = tf.keras.models.load_model(
    f"/scratch/smkang/gst_primitive/ml/v_l/name.save").get_layer("potential")

os.system(f"mkdir epp")

nat, v_s, v_l = lload()

for i in range(nat):
    with open(f"./epp/v{i+1:06d}.txt", "w") as f:
        for v in v_s[i]:
            f.write(f"{np.real(v):15.8E} {np.imag(v):15.8E}\n")
    with open(f"./epp/lv{i+1:06d}.txt", "w") as f:
        for v in v_l[i]:
            f.write(f"{v:15.8E}\n")
