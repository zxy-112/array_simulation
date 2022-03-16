# -*- coding: utf-8 -*-
# @Time    : 2021/12/1 14:50
# @Author  : zxy
# @Email   : 
# @File    : signal_cancellation.py
# @software: PyCharm

from array_simulation_initialization2 import *

SNR = [0, 10, 20, 50, 100]

name = '期望信号'
_, ax = plt.subplots(figsize=figsize, num=name)
ax.plot(t, np.real(expect_signal.signal))

for snr in SNR:
    amplitude = np.power(10, snr / 20) * noise_amplitude
    expect_signal.amplitude = amplitude
    array.add_signal({'signal': expect_signal.signal, 'theta': theta_of_expect_signal})
    array.add_signal(({'signal': s_normal_inter[0].signal, 'theta': theta_of_normal_inter[0]}))

    mvdr_weight = array.get_mvdr_weight(expect_azimuth=theta_of_expect_signal)
    response, theta = array.pattern(weights=mvdr_weight)
    name = '信噪比为' + str(snr) + 'dB下波束形成'
    fig, _ = pp1(theta, at.zxy_decibel(response), name=name)
    if save_flag:
        fig.savefig(save_path + name + suffix, **save_config)

    signal_after_beamform = np.conjugate(mvdr_weight.T) @ array.get_output()
    signal_after_beamform = signal_after_beamform.flatten()

    name = name + '后的信号波形'
    fig, ax = plt.subplots(figsize=figsize, num=name)
    ax.plot(t, np.real(signal_after_beamform))
    if save_flag:
        fig.savefig(save_path + name + suffix, **save_config)

    array.clear_signal()

plt.show()
if save_flag:
    pass
