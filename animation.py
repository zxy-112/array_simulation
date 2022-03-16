# -*- coding: utf-8 -*-
# @Time    : 2021/12/9 17:20
# @Author  : zxy
# @Email   : 
# @File    : animation.py
# @software: PyCharm

from array_simulation_initialization2 import *
import matplotlib.animation as animation

fig, ax = plt.subplots(1, 3, figsize=(12, 4))

correlation_x, correlation_y = [], []
line1, = ax[0].plot(correlation_x, correlation_y, linewidth=2)
ax[0].set_xlim((0, 0.32))
ax[0].set_ylim((0, 1))
ax[0].set_ylabel('相关系数', fontproperties=myfont)
line2, = ax[1].plot([], [])
ax[1].set_xlim((0, 10))
ax[1].set_ylim((-1.2, 1.2))
ax[1].set_xlabel('us')
ax[2].set_xlim(array.get_interest_theta())
ax[2].set_ylim((-60, 0))
ax[2].plot([theta_of_coherent_inter[0], theta_of_coherent_inter[0]], [-60, 0], color='red', linewidth=0.5, ls='--')
ax[2].plot([theta_of_expect_signal, theta_of_expect_signal], [-60, 0], color='purple', linewidth=0.5, ls='--')
ax[2].plot([theta_of_normal_inter[0], theta_of_normal_inter[0]], [-60, 0], color='orange', linewidth=0.5, ls='--')
ax[2].grid()
line3, = ax[2].plot([], [])

correlation_max = np.sum(np.power(np.abs(expect_signal.signal), 2))
delay = np.arange(0, 0.3, 0.0005)
frame_num = delay.size


def run(frame_index):

    signal1 = expect_signal.signal
    expect_signal.delay = delay[frame_index]
    signal2 = expect_signal.signal
    expect_signal.delay = 0
    correlation = np.sum(np.abs(signal1.reshape((1, -1)) @ np.conjugate(signal2.reshape((-1, 1))))) / correlation_max
    correlation_y.append(correlation)
    correlation_x.append(delay[frame_index])
    line1.set_data(correlation_x, correlation_y)

    array.clear_signal()
    array.add_signal({'signal': signal1, 'theta': theta_of_expect_signal})
    array.add_signal({'signal': signal2, 'theta': theta_of_coherent_inter[0]})
    array.add_signal({'signal': s_normal_inter[0].signal, 'theta': theta_of_normal_inter[0]})
    weights = array.get_mvdr_weight(expect_azimuth=theta_of_expect_signal)
    response, theta = array.pattern(weights=weights)
    response = at.zxy_decibel(response)
    line3.set_data(theta, response)

    signal_after_beamform = np.conjugate(weights.T) @ array.get_output()
    signal_after_beamform = np.real(signal_after_beamform.flatten())
    signal_after_beamform = signal_after_beamform / np.max(signal_after_beamform)
    line2.set_data(t, signal_after_beamform)

    return line1, line3, line2


ani_kw = dict(frames=frame_num, interval=50, blit=True, repeat=False)
ani = animation.FuncAnimation(fig, run, **ani_kw)

fig2, ax2 = plt.subplots(2, 1, gridspec_kw=gs_kw, sharex='all', sharey='all', figsize=figsize)
ax2[0].plot(t, np.real(expect_signal.signal))
line4, = ax2[1].plot([], [], color='purple')
ax2[1].set_xlabel('us')


def run2(frame_index):

    expect_signal.delay = delay[frame_index]
    signal = expect_signal.signal
    expect_signal.delay = 0
    line4.set_data(t, np.real(signal))

    return line4,


ani2 = animation.FuncAnimation(fig2, run2, **ani_kw)

ani.save(save_path + 'movie3' + suffix2)
ani2.save(save_path + 'movie4' + suffix2)

plt.show()
