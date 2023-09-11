import h5py
import numpy as np
import matplotlib.pyplot as plt

files = np.arange(0, 9900, 100)
for f in files:
    f = h5py.File("build/particles_" + str(f) + ".h5", "r")
    x = np.array(f["position"][:, 0])
    y = np.array(f["position"][:, 1])
    rho = np.array(f["density"])
    rho = np.array(f["density_acc"])
    rho = np.array(f["h"])
    print(rho)
