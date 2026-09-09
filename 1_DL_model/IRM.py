import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import cv2
import skimage.transform as skt

# dirname="datasets/EC300J-Carbon/"
# data = np.load(dirname + "projections.npy")
# ang = np.arange(-66,67,6)

dirname="datasets/OCT-Pt/"
data = np.load(dirname + "projections.npy")
ang = np.loadtxt(dirname + "Angles.txt")

os.makedirs("./rec/raw", exist_ok=True)
vol = np.zeros((data.shape[0], data.shape[1], data.shape[0]))
for i in range(data.shape[1]):
    print(i)
    tomo = skt.iradon_sart(data[:, i, :], ang)
    vol[:, :, i] = tomo

vol = vol / np.max(vol) * 255
sio.savemat("./rec/raw/rec.mat", {"vol": vol})

for i in range(data.shape[1]):
    cv2.imwrite("./rec/raw/%s.png" % i, vol[:, :, i])
    np.save("./rec/raw/%s.npy" % i, vol[:, :, i])
