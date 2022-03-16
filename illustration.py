# -*- coding: utf-8 -*-
# @Time    : 2021/10/12 15:26
# @Author  : zxy
# @Email   : 
# @File    : illustration.py
# @software: PyCharm

import numpy as np
import matplotlib.pyplot as plt
import array_tools as at

lfm = at.Lfm(bandwidth=10, pulsewidth=10)
lfm.signal_points = 10 * lfm.minimum_points
t = np.arange(lfm.signal.size) * lfm.sample_interval

gs_kw = {"hspace": 0}
fig, ax = plt.subplots(2, 1, gridspec_kw=gs_kw, figsize=(5, 8), constrained_layout=False)
ax[0].plot(np.real(lfm.signal))
lfm.delay = 2
lfm.amplitude = 10
ax[1].plot(np.real(lfm.signal))
plt.show()

lfm.delay = 0
fig2, ax2 = plt.subplots(2, 1, gridspec_kw=gs_kw, figsize=(5, 8), constrained_layout=False)
ax2[0].plot(np.real(lfm.signal))
ax2[1].plot(np.real(lfm.signal))

for item in np.hstack((ax, ax2)):
    item.set_axis_off()

# fig.savefig(r"C:\Users\ZXY\Desktop\阵列仿真\非相干.png", transparent=True)
# fig2.savefig(r"C:\Users\ZXY\Desktop\阵列仿真\相干.png", transparent=True)
