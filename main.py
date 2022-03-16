# -*- coding: utf-8 -*-
# @Time    : 2021/12/12 11:01
# @Author  : zxy
# @Email   : 
# @File    : main.py
# @software: PyCharm

import numpy as np
import matplotlib.pyplot as plt
from array_tools import Array, zxy_decibel


array_elenum = 8  # 阵元数
amplitude = [1, 1, 1, 1, 1, 1, 1, 1]  # 阵元幅度
phase = [0, 0, 0, 0, 0, 0, 0, 0]  # 阵元相位(rad)
figsize = (5, 4)


assert(len(amplitude) == array_elenum)
assert(len(phase) == array_elenum)
assert(isinstance(array_elenum, (int, np.integer)))


array = Array(ele_num=array_elenum)


weights = [a * np.exp(1j * p) for a, p in zip(amplitude, phase)]
weights = np.array(weights)
response, theta = array.pattern(weights=weights)


fig, ax = plt.subplots(2, 1, figsize=figsize)
ax[0].plot(theta, zxy_decibel(response))
ax[0].set_xlabel('angle')
ax[0].set_ylabel('dB')
ax[1].plot(theta, np.angle(response))
ax[1].set_xlabel('angle')
plt.show()
