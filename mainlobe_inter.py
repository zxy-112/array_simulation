# -*- coding: utf-8 -*-
# @Time    : 2021/12/6 14:22
# @Author  : zxy
# @Email   : 
# @File    : mainlobe_inter.py
# @software: PyCharm

from array_simulation_initialization2 import *

theta1 = 10
array.add_signal({'signal': expect_signal.signal, 'theta': theta_of_expect_signal})
array.add_signal({'signal': s_normal_inter[0].signal, 'theta': theta1})

weights = array.get_mvdr_weight(expect_azimuth=theta_of_expect_signal)
response, theta = array.pattern(weights=weights)

plot_1(theta, at.zxy_decibel(response), name='副瓣干扰', theta_of_expect_signal=theta_of_expect_signal,
       theta_of_normal_inter=theta1)

signal_after_beamform = np.conjugate(weights.T) @ array.get_output()
signal_after_beamform = signal_after_beamform.flatten()

name = '副瓣干扰MVDR波束形成后的信号波形'
plot_2(t, np.real(only_expect_signal_after_beamform), t, np.real(signal_after_beamform), name=name,
       label=('无干扰', '有干扰'))

array.clear_signal()
theta2 = 3
array.add_signal({'signal': expect_signal.signal, 'theta': theta_of_expect_signal})
array.add_signal({'signal': s_normal_inter[0].signal, 'theta': theta2})

weights = array.get_mvdr_weight(expect_azimuth=theta_of_expect_signal)
response, theta = array.pattern(weights=weights)

plot_1(theta, at.zxy_decibel(response), name='主瓣干扰', theta_of_expect_signal=theta_of_expect_signal,
       theta_of_normal_inter=theta2)

signal_after_beamform = np.conjugate(weights.T) @ array.get_output()
signal_after_beamform = signal_after_beamform.flatten()

name = '主瓣干扰MVDR波束形成后的信号波形'
plot_2(t, np.real(only_expect_signal_after_beamform), t, np.real(signal_after_beamform), name=name,
       label=('无干扰', '有干扰'))

plt.show()
