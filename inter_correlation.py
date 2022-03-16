# -*- coding: utf-8 -*-
# @Time    : 2021/12/1 11:05
# @Author  : zxy
# @Email   : 
# @File    : inter_correlation.py
# @software: PyCharm

# 干扰相干性仿真

from array_simulation_initialization2 import *
import zxy_tools as zt

delay = [0.001, 0.01, 0.05, 0.07, 0.1, 0.15]
SIR = 0
ratio = np.power(10, SIR/20)

for item in delay:
    name = '时延' + str(item) + 'us'
    fig, ax = plt.subplots(2, 1, figsize=figsize, gridspec_kw=gs_kw, num=name)

    expect_signal.delay = 0
    array.add_signal({'signal': expect_signal.signal, 'theta': theta_of_expect_signal})
    ax[0].plot(t, np.real(expect_signal.signal))

    expect_signal.delay = item
    temp = expect_signal.amplitude
    expect_signal.amplitude = expect_signal.amplitude / ratio
    array.add_signal({'signal': expect_signal.signal, 'theta': 30})
    ax[1].plot(t, np.real(expect_signal.signal), color='red')
    expect_signal.amplitude = temp

    mvdr_weight = array.get_mvdr_weight(expect_azimuth=theta_of_expect_signal)
    response, theta = array.pattern(weights=mvdr_weight)

    name = name + 'mvdr波束形成结果'
    plot_1(theta, zt.zxy_decibel(response), name=name, theta_of_coherent_inter=theta_of_coherent_inter[0])

    name = '时延' + str(item) + 'usMVDR波束形成后的信号波形'
    signal_after_beamform = np.conjugate(mvdr_weight.T) @ array.get_output()
    signal_after_beamform = signal_after_beamform.flatten()
    _, ax = plt.subplots(figsize=figsize, num=name)
    ax.plot(t, np.real(signal_after_beamform))

    array.clear_signal()

plt.show()
