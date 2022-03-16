# -*- coding: utf-8 -*-
# @Time    : 2021/11/23 11:04
# @Author  : zxy
# @Email   : 
# @File    : array_test.py
# @software: PyCharm

import numpy as np
import matplotlib.pyplot as plt
import array_tools as at

array = at.Array(16)

a_theta_1 = array.get_guide_vec(azimuth=-20)
a_theta_2 = array.get_guide_vec(azimuth=-21)
a_theta_3 = array.get_guide_vec(azimuth=-19)

cov_matrix = (1 * a_theta_1 @ np.conjugate(a_theta_1.T)
              + a_theta_2 @ np.conjugate(a_theta_2.T)
              + a_theta_3 @ np.conjugate(a_theta_3.T)
              - 0.1 * array.get_sigma_power() * np.eye(array.get_elenum(), dtype=np.complex_))
weights = np.linalg.pinv(cov_matrix) @ array.get_guide_vec(azimuth=0)

response, theta = array.pattern(weights=weights)

fig, ax = plt.subplots()
ax.plot(theta, at.zxy_decibel(response))
plt.show()
