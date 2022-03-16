# -*- coding: utf-8 -*-
# @Time    : 2021/12/3 14:14
# @Author  : zxy
# @Email   : 
# @File    : coherent_signal_cancellation.py
# @software: PyCharm

from array_simulation_initialization2 import *

array.add_signal({'signal': expect_signal.signal, 'theta': theta_of_expect_signal})
array.add_signal({'signal': s_normal_inter[0].signal, 'theta': theta_of_normal_inter[0]})
weights = array.get_mvdr_weight(expect_azimuth=theta_of_expect_signal)
response, theta = array.pattern(weights=weights)

non_coherent_signal_after_beamform = np.conjugate(weights.T) @ array.get_output()
non_coherent_signal_after_beamform = non_coherent_signal_after_beamform.flatten()

name = '无相干干扰波束形成'
plot_1(theta, at.zxy_decibel(response), name=name, theta_of_expect_signal=theta_of_expect_signal,
       theta_of_normal_inter=theta_of_normal_inter[0])

array.add_signal({'signal': s_inter[0].signal, 'theta': theta_of_coherent_inter[0]})
weights = array.get_mvdr_weight(expect_azimuth=theta_of_expect_signal)
response, theta = array.pattern(weights=weights)

coherent_signal_after_beamform = np.conjugate(weights.T) @ array.get_output()
coherent_signal_after_beamform = coherent_signal_after_beamform.flatten()

name = '有相干干扰波束形成'
plot_1(theta, at.zxy_decibel(response), name=name, theta_of_expect_signal=theta_of_expect_signal,
       theta_of_normal_inter=theta_of_normal_inter[0], theta_of_coherent_inter=theta_of_coherent_inter[0])

name = 'MVDR波束形成后的波形'
fig, ax = plt.subplots(3, 1, gridspec_kw={'hspace': 0}, figsize=(8, 6), num=name)
artist_list = []
line, = ax[0].plot(t, np.real(only_expect_signal_after_beamform), label='期望信号')
artist_list.append(line)
line, = ax[1].plot(t, np.real(non_coherent_signal_after_beamform), color='red', label='不存在相干干扰')
artist_list.append(line)
line, = ax[2].plot(t, np.real(coherent_signal_after_beamform), color='orange', label='存在相干干扰')
artist_list.append(line)
if legend_flag:
    fig.legend(handles=artist_list, prop=myfont)
# for item in ax:
#     item.set_axis_off()
for item in ax:
    item.set_yticks([])
ax[-1].set_xlabel('us')
if save_flag:
    fig.savefig(save_path + name + suffix, **save_config)

array.clear_signal()
array.add_signal({'signal': expect_signal.signal, 'theta': theta_of_expect_signal})
array.add_signal({'signal': s_normal_inter[0].signal, 'theta': theta_of_normal_inter[0]})
array.add_signal({'signal': s_normal_inter[1].signal, 'theta': theta_of_normal_inter[1]})

weights = array.get_mvdr_weight(expect_azimuth=theta_of_expect_signal)
response, theta = array.pattern(weights=weights)

name = '干扰之间相干MVDR波束形成'
plot_1(theta, at.zxy_decibel(response), name=name, theta_of_expect_signal=theta_of_expect_signal,
       theta_of_normal_inter=theta_of_normal_inter)

signal_after_beamform = np.conjugate(weights.T) @ array.get_output()
signal_after_beamform = signal_after_beamform.flatten()

name = '干扰之间相干MVDR波束形成后的信号波形'
label = ('期望信号', '存在干扰')
plot_2(t, np.real(only_expect_signal_after_beamform), t, np.real(signal_after_beamform), name=name, label=label)


plt.show()

# if save_flag:
#     plt.close('all')
