import h5py
import numpy as np
import matplotlib.pyplot as plt

f = h5py.File("build/particles_0.h5", "r")
# print(np.array(f["radius"]))
x = np.array(f["position"][:, 0])
y = np.array(f["position"][:, 1])
cos_x = np.cos(x)
sin_x = np.array(f["sin"])
sin_appr = np.array(f["sin_appr"])
print(sin_appr)
plt.plot(x, cos_x, label="real cos")
plt.plot(x, sin_x, label="real sin")
plt.scatter(x, sin_appr, label="SPH appr")
# plt.scatter(x, y)
plt.legend()
plt.show()
# plt.plot()
