# -*- coding: utf-8 -*-
# @Time    : 2021/6/30 14:50
# @Author  : zxy
# @Email   :
# @File    : array_tools.py
# @software: PyCharm
# 阵列仿真模块，主要为一个均匀线列类

import math
import numpy as np
import scipy.signal
from typing import Union


def zxy_decibel(x: np.ndarray):
    """
    将-60dB以下值改为-60dB

    :param x:
    :return:
    """
    if isinstance(x, np.ndarray):
        my_decibel = -60
        temp_arr = x.copy()
        temp_arr = np.abs(temp_arr)
        temp_arr = 20 * np.log10(temp_arr / temp_arr.max())
        temp_arr[temp_arr < my_decibel] = my_decibel
        return temp_arr
    else:
        raise TypeError


class Lfm:

    def __init__(self, bandwidth=10e6, pulsewidth=10e-6):
        """

        :param bandwidth: 带宽
        :param pulsewidth: 脉宽
        """
        self._bandwidth = bandwidth  # 信号带宽

        self._pulsewidth = pulsewidth  # 信号脉宽

        self._minimum_points = math.floor(self.pulsewidth * 2 * self.bandwidth)  # 最小采样点数

        self._signal_points = self._minimum_points  # 采样点数

        self._amplitude = 1  # 信号幅度

        self._fre_offset = 0  # 频偏

        self._signal = self._create_signal()  # 信号

        self._delay = 0  # 时延

        self._rng = np.random.default_rng()

    @property
    def bandwidth(self):
        """
        带宽
        :return:
        """
        return self._bandwidth

    @bandwidth.setter
    def bandwidth(self, value):
        if not isinstance(value, (int, float, np.number)):
            print('设置失败，参数错误')
        elif value <= 0:
            print('设置失败，参数错误')
        else:
            self._bandwidth = value
            self._update_minimum_points()
            self._update_signal()

    @property
    def pulsewidth(self):
        """
        脉宽
        :return:
        """
        return self._pulsewidth

    @pulsewidth.setter
    def pulsewidth(self, value):
        if not isinstance(value, (int, float, np.number)):
            print('设置失败，参数错误')
        elif value <= 0:
            print('设置失败，参数错误')
        else:
            self._pulsewidth = value
            self._update_minimum_points()
            self._update_signal()

    def _update_minimum_points(self):
        self._minimum_points = math.floor(self.pulsewidth * 2 * self.bandwidth)

    @property
    def minimum_points(self):
        """
        最小采样点数
        :return:
        """
        return self._minimum_points

    @property
    def signal(self):
        """
        信号
        :return:
        """
        return self._signal.copy()

    @property
    def signal_points(self):
        """
        采样点数
        :return:
        """
        return self._signal_points

    @signal_points.setter
    def signal_points(self, value):
        if isinstance(value, (int, np.integer)):
            if value >= self._minimum_points:
                self._signal_points = value
                self._update_signal()
            else:
                print('设置失败，参数错误')
        else:
            print('设置失败，参数错误')

    @property
    def amplitude(self):
        return self._amplitude

    @amplitude.setter
    def amplitude(self, value):
        flag = True
        if isinstance(value, (int, float, np.number)):
            if value > 0:
                self._amplitude = value
                self._update_signal()
                flag = False
        if flag:
            print('参数错误，设置失败')

    @property
    def fre_offset(self):
        """
        中心频率
        :return:
        """
        return self._fre_offset

    @fre_offset.setter
    def fre_offset(self, value):
        """
        设置中心频率，注意设置前需要调节采样点数防止出现欠采样
        :param value:
        """
        if isinstance(value, (int, float, np.number)):
            self._fre_offset = value
            self._update_signal()
        else:
            print('参数错误，设置失败')

    def _create_signal(self):
        t = np.linspace(-self._pulsewidth / 2, self._pulsewidth / 2, self._signal_points, dtype=np.float_)
        lfm_signal = np.exp(1j * np.pi * self._bandwidth / self._pulsewidth * np.power(t, 2)) * self._amplitude
        lfm_signal = lfm_signal * np.exp(1j * 2 * np.pi * self._fre_offset * t)
        return lfm_signal

    def _update_signal(self, from_delay: bool = False):
        if not from_delay:
            self._signal = self._create_signal()
        if self._delay:
            delay_points = np.int_(np.ceil(self._delay / self.sample_interval))
            self._signal = np.hstack((np.zeros(delay_points, dtype=np.complex_), self._signal[:-delay_points]))
        else:
            self._signal = self._create_signal()

    @property
    def sample_interval(self):
        """
        采样间隔
        :return:
        """
        return self._pulsewidth / self._signal_points

    def add_random_phase(self):
        random_phase = 2 * np.pi * (self._rng.random() - 0.5)
        self._signal = self._signal * np.exp(1j * random_phase)
        return random_phase

    def add_phase(self, phase: float):
        self._signal = self._signal * np.exp(1j * phase)

    @property
    def delay(self):
        return self._delay

    @delay.setter
    def delay(self, value):
        if isinstance(value, (int, float, np.number)):
            if 0 <= value < self._pulsewidth:
                self._delay = value
                self._update_signal(from_delay=True)
            else:
                print('参数错误，设置失败')
        else:
            print('参数错误，设置失败')

    @property
    def t(self):
        return np.linspace(0, self._pulsewidth, self._signal_points, dtype=np.float_)


class Array:

    def __init__(self, ele_num: int):

        if ele_num <= 0 or not isinstance(ele_num, (int, np.integer)):
            raise ValueError
        self._elenum = ele_num  # 阵元数目

        self._d_lmd_ratio = 0.5  # 阵元间距与波长的比值，初始为0.5

        self._lmd = 1.  # 波长，可通过setlmd设置

        self._struct = self._calculate_struct()  # 阵列结构

        self._d = self._lmd * self._d_lmd_ratio  # 默认阵元间隔为半个波长，可通过setratio设置

        self._position = self._calculate_position()  # 阵元位置向量

        self._interest_theta = (-90., 90.)  # 感兴趣的角度区域

        self._points = self._calculate_points()  # 感兴趣的角度点数

        self._theta = np.linspace(self._interest_theta[0], self._interest_theta[1], self._points)  # 感兴趣的角度矢量

        self._manifold = self._calculate_manifold()  # 阵列流型

        self._expect_azimuth = 0.  # 期望信号角度

        self._default_weights = self._calculate_default_weights()  # 计算默认权矢量

        self._signal = []  # 输入信号列表，通过addsignal添加

        self._flag = False

        self._signal_len_default = 1024

        self._signal_len = self._signal_len_default  # 输入信号长度

        self._sigma_power = 1  # 噪声方差

        self._noise = self._create_noise()  # 噪声快拍构成的矩阵，快拍数为信号长度

        self._output = self._calculate_output()  # 阵列输出

        self._covmat = self._output @ np.conjugate(self._output.T) / self._output.shape[1]  # 协方差矩阵

    def set_elenum(self, elenum: int):
        """
        改变阵元数量

        :param elenum: 阵元个数
        """
        if isinstance(elenum, (int, np.integer)):

            # region 改变阵元数量同时调整阵列结构和阵元位置矢量
            if elenum >= 2:
                self._elenum = elenum
                self._update_struct()
                self._update_position()
                self._update_noise()
            # endregion

            else:
                print('输入错误，参数设置失败')
        else:
            print('输入错误，参数设置失败')

    def get_elenum(self):
        """
        返回阵元数量

        :return:
        """
        return self._elenum

    def set_ratio(self, ratio: float = 0.5):
        """
        设置阵元间距与波长的比值

        :param ratio:
        """
        if isinstance(ratio, (float, np.floating)):

            # region 设置比值同时调整阵列结构和阵元间距
            if ratio > 0:
                self._d_lmd_ratio = ratio
                self._update_struct()
                self._update_d()
            # endregion

            else:
                print('输入错误，参数设置失败')

        else:
            print('输入错误，参数设置失败')

    def get_ratio(self):
        """
        返回阵元间隔与波长的比值
        :return:
        """
        return self._d_lmd_ratio

    def set_lmd(self, lmd: float):
        """
        设置工作波长
        :param lmd:
        """
        if isinstance(lmd, (float, int, np.number)):

            # region 设置波长同时改变阵元间距
            if lmd > 0:
                self._lmd = lmd
                self._update_d()
            # endregion

            else:
                print('输入错误，参数设置失败')
        else:
            print('输入错误，参数设置失败')

    def get_lmd(self):
        """
        返回波长
        :return:
        """
        return self._lmd

    def _calculate_struct(self):
        """
        计算阵列结构
        :return:
        """
        struct_array = np.arange(0, self._elenum) * self._d_lmd_ratio
        return struct_array

    def _update_struct(self):
        """
        更新阵列结构
        """
        # region 更新阵列结构，同时更新阵列流型和默认权矢量
        self._struct = self._calculate_struct()
        self._update_manifold()
        self._update_default_weights()
        self._update_output()
        # endregion

    def get_struct(self):
        """
        返回阵列结构
        :return:
        """
        return self._struct.copy()

    def _update_d(self):
        """
        更新阵元间距
        """
        # region 更新阵元间距同时更新阵元位置向量
        self._d = self._lmd * self._d_lmd_ratio
        self._update_position()
        # endregion

    def get_d(self):
        """
        返回阵元间隔
        :return:
        """
        return self._d

    def _calculate_position(self):
        """
        计算阵元位置
        :rtype: np.ndarray
        :return:
        """
        position_array = np.arange(0, self._elenum) * self._d
        return position_array

    def _update_position(self):
        # region 更新阵元位置向量
        self._position = self._calculate_position()
        # endregion

    def get_position(self):
        """
        获取阵元位置向量
        :return:
        """
        return self._position.copy()

    def set_interest_theta(self, theta: tuple = (-90., 90.)):
        if isinstance(theta, tuple) and len(theta) == 2:
            if -90 < theta[0] < theta[1] < 90:
                # region 设置感兴趣的角度同时更新角度个数和角度矢量
                self._interest_theta = theta
                self._update_points()
                # endregion
            else:
                print('输入错误，参数设置失败')
        else:
            print('输入错误，参数设置失败')

    def get_interest_theta(self):
        """
        返回感兴趣的角度区域
        :return:
        """
        return self._interest_theta

    def _calculate_points(self):
        """
        计算合适的角度个数
        :return:
        """
        theta_begin = int(self._interest_theta[0])
        theta_end = int(self._interest_theta[1])
        theta_points = (theta_end - theta_begin) * 10 + 1
        return theta_points

    def _update_points(self):
        """
        更新感兴趣的角度个数
        """

        # region 更新角度数量同时更新角度矢量
        self._points = self._calculate_points()
        self._update_theta()
        # endregion

    def set_points(self, points: int):
        """
        设置感兴趣的角度个数
        :param points:
        """

        if isinstance(points, (int, np.integer)):
            if points > 0:

                # region 改变感兴趣的角度个数同时更新角度矢量
                self._points = points
                self._update_theta()
                # endregion

            else:
                print('输入错误，参数设置失败')
        else:
            print('输入错误，参数设置失败')

    def get_points(self):
        return self._points

    def _update_theta(self):
        # region 更新角度矢量同时更新阵列流型
        self._theta = np.linspace(self._interest_theta[0], self._interest_theta[1], self._points)
        self._update_manifold()
        # endregion

    def get_theta(self):
        return self._theta.copy()

    def _calculate_manifold(self):
        """
        计算阵列流型
        :return:
        """
        arr1 = 2 * np.pi * self._struct
        arr2 = np.sin(np.deg2rad(self._theta))
        arr_temp = np.exp(1j * np.outer(arr1, arr2))
        return arr_temp

    def _update_manifold(self):
        """
        更新阵列流型
        """
        self._manifold = self._calculate_manifold()

    def get_manifold(self):
        return self._manifold.copy()

    def set_expect_azimuth(self, azimuth: float = 0.):
        """
        设置期望方向
        :param azimuth:
        """
        if isinstance(azimuth, (float, int, np.number)):
            if -90 <= azimuth <= 90:
                # region 设置期望方向，并更新默认权矢量
                self._expect_azimuth = azimuth
                self._update_default_weights()
                # endregion
            else:
                print('参数错误，设置失败')
        else:
            print('输入错误，参数设置失败')

    def get_expect_azimuth(self):
        return self._expect_azimuth

    def _calculate_default_weights(self):
        """
        计算默认权矢量
        """
        return np.exp(1j * 2 * np.pi * self._struct *
                      np.sin(np.deg2rad(self._expect_azimuth))).reshape((-1, 1)) / self._struct.size

    def _update_default_weights(self):
        self._default_weights = self._calculate_default_weights()

    def get_default_weights(self):
        return self._default_weights.copy()

    def add_signal(self, signal: dict):
        """
        给阵列输入信号，signal字典中必须有signal和theta两个key键
        :param signal:
        """
        if isinstance(signal, dict):
            if 'signal' in signal and 'theta' in signal:
                if isinstance(signal['signal'], np.ndarray):
                    if -90 <= signal['theta'] <= 90:
                        if not self._flag:
                            signal_copy = signal.copy()
                            signal_copy['signal'] = signal_copy['signal'].flatten()
                            self._signal.append(signal_copy)
                            self._flag = not self._flag
                            self._update_signal_len()
                        else:
                            if signal['signal'].size == self._signal_len:
                                signal_copy = signal.copy()
                                signal_copy['signal'] = signal_copy['signal'].flatten()
                                self._signal.append(signal_copy)
                                self._update_output()
                            else:
                                print('输入的所有信号的长度必须一致，信号添加失败')
                    else:
                        print('输入错误，参数设置失败')
                else:
                    print('输入错误，参数设置失败')
            else:
                print('传入的信号字典必须包括signal和theta两个参数，信号添加失败')
        else:
            print('输入错误，参数设置失败')

    def clear_signal(self):
        """
        清除所有输入信号
        """
        if self._flag:
            self._signal = []
            self._flag = False
            self._update_signal_len()

    def get_signal(self):
        return self._signal.copy()

    def get_flag(self):
        """
        判断是否有信号输入
        :return:
        """
        return self._flag

    def _update_signal_len(self):
        """
        更新信号长度
        """
        if not self._flag:
            self._signal_len = self._signal_len_default
        else:
            self._signal_len = self._signal[0]['signal'].size
        self._update_noise()

    def get_signal_len(self):
        return self._signal_len

    def set_sigma_power(self, sigma_power: float):
        """
        设置噪声方差
        :param sigma_power:
        """
        if isinstance(sigma_power, (float, int, np.number)):
            if sigma_power > 0:
                # region 同时改变噪声信号
                self._sigma_power = sigma_power
                self._update_noise()
                # endregion
            else:
                print('输入错误，参数设置失败')
        else:
            print('输入错误，参数设置失败')

    def get_sigma_power(self):
        return self._sigma_power

    def _create_noise(self, seed=1024):
        """
        产生噪声
        :return:
        """
        rng = np.random.default_rng(seed=seed)
        noise = (rng.standard_normal((self._elenum, self._signal_len)) * np.sqrt(self._sigma_power / 2) +
                 1j * rng.standard_normal((self._elenum, self._signal_len)) * np.sqrt(self._sigma_power / 2))
        return noise

    def _update_noise(self):
        """
        更新噪声
        """
        # region 同时更新阵列输出信号
        self._noise = self._create_noise()
        self._update_output()
        # endregion

    def get_noise(self):
        return self._noise.copy()

    def _calculate_output(self):
        """
        根据输入信号和噪声信号计算阵列输出信号
        """
        if not self._flag:
            return self.get_noise()
        else:
            intermediate_result = self.get_noise()  # 存放中间结果
            for item in self._signal:
                temp_arr = np.exp(1j * 2 * np.pi * self._struct * np.sin(np.deg2rad(item['theta'])))  # 计算导向矢量
                temp_arr = temp_arr.reshape((-1, 1))
                intermediate_result = intermediate_result + np.kron(temp_arr, item['signal'])
            return intermediate_result

    def _update_output(self):
        """
        更新阵列输出
        """
        # region 更新阵元个数，先更新了阵列结构，在没有更新噪声信号时可能出现维数不匹配
        try:
            self._output = self._calculate_output()
            self._update_covmat()
        except ValueError:
            pass
        # endregion

    def get_output(self):
        """
        获取阵列输出信号
        :return:
        """
        return self._output.copy()

    def _update_covmat(self):
        """
        更新协方差矩阵
        """
        self._covmat = self._output @ np.conjugate(self._output.T) / self._output.shape[1]

    def get_covmat(self, snapshots=None):
        """
        返回协方差矩阵
        :param snapshots: 用多少快拍数来获得协方差矩阵
        :return:
        """
        if snapshots is None:
            snapshots = self._output.shape[1]
        elif isinstance(snapshots, (int, np.integer)):
            if snapshots > self._output.shape[1]:
                snapshots = self._output.shape[1]
            elif snapshots <= 0:
                raise ValueError
            else:
                pass
        else:
            raise TypeError

        data_matrix = self._output[:, 0:snapshots]
        covmat = data_matrix @ np.conjugate(data_matrix.T) / snapshots

        return covmat.copy()

    # region 波束形成

    def pattern(self, expect_azimuth: float = 0., split_flag=True, **kwargs):
        """
        返回方向图，可给定权矢量weights = weights
        :return:
        :param split_flag:
        :param expect_azimuth:
        :param kwargs:
        """
        if isinstance(expect_azimuth, (float, int, np.number)):
            if -90 <= expect_azimuth <= 90 and expect_azimuth != self._expect_azimuth:
                self.set_expect_azimuth(expect_azimuth)
            elif expect_azimuth == self._expect_azimuth:
                pass
            else:
                print('输入错误，参数设置失败')
        else:
            raise TypeError

        if 'weights' in kwargs:
            weights = kwargs['weights']
            # region 判断
            if isinstance(kwargs['weights'], np.ndarray):
                if kwargs['weights'].size != self._elenum:
                    raise ValueError
                else:
                    pass
            else:
                raise TypeError
            # endregion
            weights = np.reshape(weights, (-1, 1))
        else:
            weights = self.get_default_weights()

        response = (np.conjugate(weights.T) @ self._manifold).flatten()
        theta = self._theta.copy()

        if 'extra_theta' in kwargs:
            extra_theta = kwargs['extra_theta']
            extra_theta = [item for item in extra_theta]
            extra_response = [(weights @ self.get_guide_vec(azimuth=theta)).flatten()[0] for theta in extra_theta]

            extra_theta = np.array(extra_response, dtype=np.float_)
            extra_response = np.array(extra_response, dtype=np.complex_)

            if split_flag:
                return response, theta, extra_response, extra_theta
            else:
                return np.hstack((response, extra_response)), np.hstack((theta, extra_theta))

        return response, theta

    def get_guide_vec(self, azimuth: float = -100.):
        """
        获得给定方向的导向矢量
        :param azimuth:
        :return:
        """

        if isinstance(azimuth, (float, int, np.number)):
            if azimuth == -100.:
                azimuth = self._expect_azimuth
            elif -90 <= azimuth <= 90:
                pass
            else:
                raise ValueError
        else:
            raise TypeError

        return np.exp(1j * 2 * np.pi * self._struct *
                      np.sin(np.deg2rad(azimuth))).reshape((-1, 1))

    def get_mvdr_weight(self, expect_azimuth: float = 0.):
        """
        返回MVDR准则下的权矢量
        :param expect_azimuth:
        :return:
        """
        if isinstance(expect_azimuth, (float, int, np.number)):
            if expect_azimuth < -90 or expect_azimuth > 90:
                raise ValueError
        else:
            raise ValueError
        guide_vec = self.get_guide_vec(expect_azimuth)
        inv_covmat = np.linalg.pinv(self._covmat)
        return inv_covmat @ guide_vec / np.abs(np.conjugate(guide_vec.T) @ inv_covmat @ guide_vec)

    def get_mcmv_weight(self, expect_azimuth: float = 0., coherent_inter_azimuth=None):
        """
        返回多约束最小方差算法波束形成的权矢量
        :param expect_azimuth:
        :param coherent_inter_azimuth:
        :return:
        """
        if isinstance(coherent_inter_azimuth, (list, tuple, np.ndarray)):
            matrix_ac = np.empty((self._elenum, len(coherent_inter_azimuth) + 1), dtype=np.complex_)
            matrix_ac[:, 0:0 + 1] = self.get_guide_vec(azimuth=expect_azimuth).reshape(matrix_ac[:, 0:0 + 1].shape)
            for k, item in enumerate(coherent_inter_azimuth):
                matrix_ac[:, k + 1:k + 2] = self.get_guide_vec(azimuth=item).reshape((self._elenum, 1))
            f = np.vstack((np.array([[1]], dtype=np.complex_),
                           np.zeros((len(coherent_inter_azimuth), 1), dtype=np.complex_)))
        elif coherent_inter_azimuth is None:
            matrix_ac = self.get_guide_vec(azimuth=expect_azimuth).reshape((self._elenum, -1))
            f = np.array([[1]], dtype=np.complex_)
        else:
            raise TypeError

        inv_covmat = np.linalg.pinv(self._covmat)
        weights = inv_covmat @ matrix_ac @ np.linalg.pinv(np.conjugate(matrix_ac.T) @ inv_covmat @ matrix_ac) @ f
        return weights

    def get_matrix_ac(self, expect_azimuth, coherent_inter_azimuth):
        """
        返回CTMV，CTP等方法中用到的A_c矩阵
        :param expect_azimuth:
        :param coherent_inter_azimuth:
        :return:
        """
        if isinstance(coherent_inter_azimuth, (list, tuple)):
            matrix_ac = np.empty((self._elenum, len(coherent_inter_azimuth) + 1), dtype=np.complex_)
            matrix_ac[:, 0:0 + 1] = self.get_guide_vec(azimuth=expect_azimuth).reshape(matrix_ac[:, 0:0 + 1].shape)
            for k, item in enumerate(coherent_inter_azimuth):
                matrix_ac[:, k + 1:k + 2] = self.get_guide_vec(azimuth=item).reshape((self._elenum, 1))
        elif coherent_inter_azimuth is None:
            matrix_ac = self.get_guide_vec(azimuth=expect_azimuth).reshape((self._elenum, -1))
        else:
            raise TypeError

        return matrix_ac

    def get_ctmv_weight(self, expect_azimuth: float = 0., coherent_inter_azimuth=None, lmd=0):
        """
        返回辅助变换最小方差算法权矢量
        :param expect_azimuth: 期望信号角度，默认为零度
        :param coherent_inter_azimuth: 相干干扰角度，需可迭代
        :param lmd: 对角加载量
        :return: 列向量
        """
        if isinstance(coherent_inter_azimuth, (list, tuple)):
            matrix_ac = np.empty((self._elenum, len(coherent_inter_azimuth) + 1), dtype=np.complex_)
            matrix_ac[:, 0:0 + 1] = self.get_guide_vec(azimuth=expect_azimuth).reshape(matrix_ac[:, 0:0 + 1].shape)
            for k, item in enumerate(coherent_inter_azimuth):
                matrix_ac[:, k + 1:k + 2] = self.get_guide_vec(azimuth=item).reshape((self._elenum, 1))
            matrix_bc = matrix_ac.copy()
            matrix_bc[:, 0:1] = 0
        elif coherent_inter_azimuth is None:
            matrix_ac = self.get_guide_vec(azimuth=expect_azimuth).reshape((self._elenum, -1))
            matrix_bc = np.zeros((self._elenum, 1), dtype=np.complex_)
        else:
            raise TypeError

        cov_matrix = self._covmat
        cov_matrix_lmd = cov_matrix + lmd * np.eye(self._elenum, dtype=np.complex_)  # 对角加载
        inv_covmat = np.linalg.pinv(cov_matrix_lmd)

        # region 线性变换矩阵
        transform_matrix = (np.eye(self._elenum, dtype=np.complex_)
                            - (matrix_ac - matrix_bc)
                            @ np.linalg.pinv(np.conjugate(matrix_ac.T) @ inv_covmat @ matrix_ac)
                            @ np.conjugate(matrix_ac.T) @ inv_covmat)
        # endregion

        # region 噪声特性恢复
        cov_matrix = (transform_matrix @ self._covmat @ np.conjugate(transform_matrix.T)
                      - self._sigma_power * transform_matrix @ np.conjugate(transform_matrix.T)
                      + self._sigma_power * np.eye(self._elenum, dtype=np.complex_))
        # endregion

        inv_covmat = np.linalg.pinv(cov_matrix)
        a_theta_0 = self.get_guide_vec(azimuth=expect_azimuth)
        weights = (inv_covmat @ a_theta_0) / (np.conjugate(a_theta_0.T) @ inv_covmat @ a_theta_0)
        return weights

    def get_my_ctmv_weight(self, expect_azimuth: float = 0., coherent_inter_azimuth=None, lmd=0, matrix_e=None):
        """
        返回辅助变换最小方差算法权矢量
        :param expect_azimuth:
        :param coherent_inter_azimuth: 相干干扰角度，需可迭代
        :param lmd: 对角加载量
        :param matrix_e:
        :return: 列向量
        """
        if isinstance(coherent_inter_azimuth, (list, tuple)):
            matrix_ac = np.empty((self._elenum, len(coherent_inter_azimuth) + 1), dtype=np.complex_)
            matrix_ac[:, 0:0 + 1] = self.get_guide_vec(azimuth=expect_azimuth).reshape(matrix_ac[:, 0:0 + 1].shape)
            for k, item in enumerate(coherent_inter_azimuth):
                matrix_ac[:, k + 1:k + 2] = self.get_guide_vec(azimuth=item).reshape((self._elenum, 1))
            matrix_bc = matrix_ac.copy()
            matrix_bc[:, 0:1] = 0
        elif coherent_inter_azimuth is None:
            matrix_ac = self.get_guide_vec(azimuth=expect_azimuth).reshape((self._elenum, -1))
            matrix_bc = np.zeros((self._elenum, 1), dtype=np.complex_)
        else:
            raise TypeError

        if matrix_e is None:
            raise ValueError
        else:
            matrix_e = matrix_e.reshape((1, -1))

        a_theta_0 = self.get_guide_vec(azimuth=expect_azimuth)
        cov_matrix = self._covmat
        cov_matrix_lmd = cov_matrix + lmd * np.eye(self._elenum, dtype=np.complex_)  # 对角加载
        inv_covmat = np.linalg.pinv(cov_matrix_lmd)

        # region 线性变换矩阵
        transform_matrix = (np.eye(self._elenum, dtype=np.complex_)
                            - a_theta_0 @ matrix_e @ np.conjugate(matrix_ac.T) @ inv_covmat
                            - ((matrix_ac - matrix_bc) @ np.linalg.pinv(np.conjugate(matrix_ac.T)
                                                                        @ inv_covmat @ matrix_ac) - 2 * a_theta_0 @
                               matrix_e)
                            @ np.conjugate(matrix_ac.T) @ inv_covmat)
        # endregion

        # region 噪声特性恢复
        cov_matrix = (transform_matrix @ self._covmat @ np.conjugate(transform_matrix.T)
                      - self._sigma_power * transform_matrix @ np.conjugate(transform_matrix.T)
                      + self._sigma_power * np.eye(self._elenum, dtype=np.complex_))
        # endregion

        inv_covmat = np.linalg.pinv(cov_matrix)
        weights = (inv_covmat @ a_theta_0) / (np.conjugate(a_theta_0.T) @ inv_covmat @ a_theta_0)
        return weights
        pass

    def get_ctp_weight(self, expect_azimuth: float = 0., coherent_inter_azimuth=None, lmd=0):
        """
        返回变换投影法的权矢量
        :param expect_azimuth: 期望信号角度
        :param coherent_inter_azimuth: 相干干扰角度，需可迭代
        :param lmd: 对角加载量
        :return: 列向量
        """
        matrix_ac = self.get_matrix_ac(expect_azimuth, coherent_inter_azimuth)  # 构建矩阵Ac

        cov_matrix = self._covmat
        cov_matrix_lmd = cov_matrix + lmd * np.eye(self._elenum, dtype=np.complex_)  # 对角加载
        inv_covmat = np.linalg.pinv(cov_matrix_lmd)

        # region 变换矩阵
        transform_matrix = (np.eye(self._elenum, dtype=np.complex_)
                            - matrix_ac
                            @ np.linalg.pinv(np.conjugate(matrix_ac.T) @ inv_covmat @ matrix_ac)
                            @ np.conjugate(matrix_ac.T) @ inv_covmat)
        # endregion

        # region 噪声特性恢复
        cov_mat = (transform_matrix @ self._covmat @ np.conjugate(transform_matrix.T)
                   - self._sigma_power * transform_matrix @ np.conjugate(transform_matrix.T)
                   + self._sigma_power * np.eye(self._elenum))
        # endregion

        u, s, _ = np.linalg.svd(cov_mat, hermitian=True)  # 非相干干扰和噪声协防差矩阵的特征值分解

        col_index = -1
        for item in s:
            if item >= 2 * self._sigma_power:
                col_index = col_index + 1
                continue
            else:
                break

        if col_index == -1:
            normal_inter_signal_projection_matrix = np.zeros((self._elenum, self._elenum), dtype=np.complex_)
        else:
            matrix_u = u[:, :col_index + 1]
            normal_inter_signal_projection_matrix = matrix_u @ np.conjugate(matrix_u.T)
        noise_space_projection_matrix = np.eye(self._elenum, dtype=np.complex_) - normal_inter_signal_projection_matrix
        # 噪声空间的正交投影矩阵

        u2, _, _ = np.linalg.svd(self._covmat)
        main_eig_vec = u2[:, 0:1]  # 协方差矩阵的主特征向量

        opt_weight = noise_space_projection_matrix @ main_eig_vec

        return opt_weight

    def get_spt_weight(self, expect_azimuth: float = 0., coherent_inter_azimuth=None, lmd=0):

        matrix_ac = self.get_matrix_ac(expect_azimuth=expect_azimuth, coherent_inter_azimuth=coherent_inter_azimuth)
        matrix_bc = matrix_ac.copy()
        matrix_bc[:, 0:1] = -matrix_bc[:, 0:1]

        cov_matrix = self._covmat
        cov_matrix_lmd = cov_matrix + lmd * np.eye(self._elenum, dtype=np.complex_)  # 对角加载
        inv_covmat = np.linalg.pinv(cov_matrix_lmd)

        # region 线性变换矩阵
        transform_matrix = (np.eye(self._elenum, dtype=np.complex_)
                            - (matrix_ac - matrix_bc)
                            @ np.linalg.pinv(np.conjugate(matrix_ac.T) @ inv_covmat @ matrix_ac)
                            @ np.conjugate(matrix_ac.T) @ inv_covmat)
        transform_matrix = (transform_matrix + np.eye(transform_matrix.shape[0], dtype=transform_matrix.dtype)) / 2
        # endregion

        # region 噪声特性恢复
        cov_matrix = (transform_matrix @ self._covmat @ np.conjugate(transform_matrix.T)
                      - self._sigma_power * transform_matrix @ np.conjugate(transform_matrix.T)
                      + self._sigma_power * np.eye(self._elenum, dtype=np.complex_))
        # endregion

        inv_covmat = np.linalg.pinv(cov_matrix)
        a_theta_0 = self.get_guide_vec(azimuth=expect_azimuth)
        spt_weight = (inv_covmat @ a_theta_0) / (np.conjugate(a_theta_0.T) @ inv_covmat @ a_theta_0)

        return spt_weight

    # endregion

    def spatial_smoothing(self, num, expect_azimuth):

        if isinstance(num, (int, np.integer)) and isinstance(expect_azimuth, (int, float, np.number)):
            pass
        else:
            raise TypeError

        cov_mat = np.zeros((num, num), dtype=np.complex_)
        iteration_times = self._elenum - num + 1

        for k in range(iteration_times):
            output = self._output[k:k+num, :]
            cov_mat = cov_mat + np.dot(output, np.conjugate(output.T)) / num

        cov_mat = cov_mat / iteration_times
        guide_vec = self.get_guide_vec(azimuth=expect_azimuth)[:num, :]
        weights = np.dot(np.linalg.pinv(cov_mat), guide_vec)

        pattern = np.dot(np.conjugate(weights.T), self._manifold[:num, :]).flatten()
        theta = self._theta.copy()
        output_signal = np.dot(np.conjugate(weights.T), self._output[:num, :]).flatten()

        smooth_result = {'weights': weights,
                         'output_signal': output_signal,
                         'pattern': pattern,
                         'theta': theta}

        return smooth_result

    # region 空间谱估计

    def fft_spectrum_evaluation(self, length: int = 8, begin: int = 0):
        """
        用fft估计空间谱
        :param begin: 开始的快拍数
        :param length: 快拍数（积累数）
        :return:
        """
        if isinstance(length, (int, np.integer)) and isinstance(begin, (int, np.integer)):
            pass
        else:
            raise ValueError

        # region 加窗后取某几个快拍做fft
        snapshot_matrix = self._output[:, begin: begin + length]
        window_seq = scipy.signal.get_window('hamming', snapshot_matrix.shape[0]).reshape((-1, 1))  # 窗函数
        window_matrix = np.tile(window_seq, (1, snapshot_matrix.shape[1]))
        snapshot_matrix = snapshot_matrix * window_matrix  # 加窗
        fft_length = 2 ** np.asarray(np.ceil(np.log2(snapshot_matrix.shape[0])), dtype=np.int_) * (2 ** 4)  # fft长度
        fft_result = np.fft.fftshift(np.fft.fft(snapshot_matrix, fft_length, axis=0), axes=0)  # 纵向做fft
        fft_result = np.fft.fft(fft_result, axis=1)  # 横向做fft，积累8个
        # endregion

        # region 取出最大值对应的列
        max_index = np.unravel_index(np.argmax(np.abs(fft_result)), fft_result.shape)
        fft_result = fft_result[:, max_index[1]]
        # endregion

        # region 产生角度序列
        freq = np.fft.fftshift(np.fft.fftfreq(fft_length) * 2 * np.pi)  # 频率序列
        theta = np.rad2deg(np.arcsin(freq / (2 * np.pi * self._d_lmd_ratio)))  # 角度序列
        # endregion

        return fft_result, theta

    def music_spectrum_evaluation(self):
        pass

    def bartlett_spectrum_evaluation(self, array_output: Union[np.ndarray, None] = None):
        """
        常规波束形成方法实现波达方向估计
        """
        manifold_matrix = self.get_manifold()  # 阵列流型
        if not (array_output is None):
            cov_mat = np.matmul(array_output, np.conjugate(array_output.T)) / self._elenum
        else:
            cov_mat = self.get_covmat()  # 协方差矩阵
        length = manifold_matrix.shape[1]  # 所有导向矢量的数量
        space_spectrum = np.empty(length, dtype=np.complex_)  # 用于存储谱估计的结果
        theta = self.get_theta()  # 角度序列

        for index in range(length):
            weight = manifold_matrix[:, index].reshape((-1, 1))  # 权矢量
            space_spectrum[index] = (np.conjugate(weight).T @ cov_mat @ weight).flatten()[0]  # 逐一求每个方位的输出功率
        space_spectrum = np.real(space_spectrum)  # 计算结果为复数，转换为实数

        return space_spectrum, theta

    def mv_spectrum_evaluation(self):
        """
        最小方差波束扫描方法实现波达方向估计
        """
        manifold_matrix = self.get_manifold()  # 阵列流型
        cov_mat = self.get_covmat()  # 协方差矩阵
        length = manifold_matrix.shape[1]  # 所有导向矢量的数量
        space_spectrum = np.empty(length, dtype=np.complex_)  # 用于存储谱估计的结果
        theta = self.get_theta()  # 角度序列

        for index in range(length):
            weight = manifold_matrix[:, index].reshape((-1, 1))  # 导向矢量
            space_spectrum[index] = (np.conjugate(weight).T @
                                     np.linalg.pinv(cov_mat) @ weight).flatten()[0]
        space_spectrum = 1 / np.abs(space_spectrum)  # 计算结果为复数，转换为实数，并取倒数

        return space_spectrum, theta

    # endregion


if __name__ == '__main__':
    array1 = Array(4)
