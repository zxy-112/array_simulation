# -*- coding: utf-8 -*-
# @Time    : 2021/10/12 15:26
# @Author  : zxy
# @Email   : 
# @File    : zxy_tools.py
# @software: PyCharm
import numpy as np


def _int2bin(num: int, length: int = 1):
    # region 确定正负
    if num >= 0:
        flag = ''
    else:
        flag = '-'
    # endregion

    # region 除二取余
    temp_var = []
    num = abs(num)
    while num != 0:
        temp_var.append(str(num % 2))
        num = num // 2
    # endregion

    # region 获得二进制构成的字符串
    temp_var = temp_var[::-1]
    temp_var = ''.join(temp_var)
    # endregion

    len_temp = len(temp_var)
    if length > len_temp:
        return flag + '0b' + '0' * (length - len_temp) + temp_var
    else:
        return flag + '0b' + temp_var


def _is_even_power(arr: np.ndarray):
    length = arr.size
    if length == 1:
        return False
    while length > 1:
        remainder = length % 2
        if remainder:
            return False
        length = length // 2
    return True


def is_even(num: int):
    if num % 2 == 0:
        return True
    else:
        return False


def _reverse_order(arr_size: int):
    """
    fft前的抽序算法
    :param arr_size:
    :return:
    """
    arr_index = list(range(arr_size))
    max_len = len(_int2bin(arr_index[-1])) - 2
    for k, v in enumerate(arr_index):
        var1 = _int2bin(v, max_len)
        arr_index[k] = eval('0b' + var1[:1:-1])
    return arr_index


def _fft_even(arr: np.ndarray):
    """
    返回偶数长度array的fft
    :param arr:
    """
    # region 插序后的序列
    temp_arr = arr.copy()
    length = temp_arr.size
    temp_arr[:] = temp_arr[_reverse_order(length)]
    # endregion

    n = int(np.round(np.log2(length)))  # 蝶形算法次数
    for k in range(n):
        var1 = 2 ** k
        lis1 = list(range(var1))
        # region 旋转因子
        rotation_factor = np.tile(np.exp(-1j * 2 * (2 ** (n - k - 1)) *
                                         np.pi / length * np.arange(var1, dtype=complex)),
                                  round(length / 2 / var1))
        # endregion
        index_1 = []
        index_2 = []
        for k_2 in range(length):
            if k_2 % (2 ** (k + 1)) in lis1:
                index_1.append(k_2)
            else:
                index_2.append(k_2)
        # print(len(index_1),len(index_2))
        temp_arr[index_2] = temp_arr[index_2] * rotation_factor
        for k_2, k_3 in zip(index_1, index_2):
            temp_arr[k_2], temp_arr[k_3] = temp_arr[k_2] + temp_arr[k_3], temp_arr[k_2] - temp_arr[k_3]
    return temp_arr


def _ifft_even(arr: np.ndarray):
    """
    返回偶数长度array的ifft
    :param arr:
    :return:
    """
    var1 = _fft_even(arr)
    var2 = np.concatenate((var1[0:1], var1[-1:0:-1]))
    var2 = var2 / len(arr)
    return var2


def czt(arr, a_0=1, theta_0=0, points=-1, delta_theta=-1, w_0=1):
    """
    计算czt，可以用来计算任意点数的fft
    :param arr:
    :param a_0:
    :param theta_0:
    :param points:
    :param delta_theta:归一化频率采样间隔
    :param w_0:
    :return:
    """
    length = arr.size
    if points == -1:
        points = length
        delta_theta = 2 * np.pi / points
    a = a_0 * np.exp(1j * theta_0)
    w = w_0 * np.exp(-1j * delta_theta)
    var1 = np.arange(length, dtype=complex)
    var2 = np.power(a, -var1)  # a^{-n}
    var3 = np.power(var1, 2)  # n^2
    var4 = np.power(w, var3 / 2)  # w^{n^2/2}
    var5 = arr * var2 * var4  # g(k)
    var6 = np.arange(-length + 1, points, dtype=complex)  # n
    var7 = np.power(var6, 2)  # n^2
    var8 = np.power(w, -var7 / 2)  # h(k)
    length_fft = np.power(2, np.ceil(np.log2(length + points - 1)))
    length_fft = np.asarray(length_fft, dtype=np.int64)
    var5 = np.concatenate((var5, np.zeros((length_fft - len(var5),),
                                          dtype=complex)))
    var8 = np.concatenate((var8, np.zeros((length_fft - len(var8),),
                                          dtype=complex)))
    var5 = _fft_even(var5)
    var8 = _fft_even(var8)
    var9 = var5 * var8
    var9 = _ifft_even(var9)
    var9 = var9[length - 1:length - 1 + points]
    var10 = np.arange(points, dtype=complex)
    var11 = np.power(var10, 2)
    var12 = np.power(w, var11 / 2)
    var13 = var12 * var9
    return var13


def fft(arr):
    """
    计算任意长度序列的fft
    :param arr:
    :return:
    """
    if _is_even_power(arr):
        return _fft_even(arr)
    else:
        return czt(arr)


def ifft(arr):
    """
    计算任意长度序列的ifft
    :param arr:
    :return:
    """
    var1 = fft(arr)
    var2 = np.concatenate((var1[0:1], var1[-1:0:-1]))
    var2 = var2 / len(arr)
    return var2


def zxy_decibel(x: np.ndarray):
    """
    将-40dB以下值改为-40dB

    :param x:
    :return:
    """
    if isinstance(x, np.ndarray):
        my_decibel = -100
        temp_arr = x.copy()
        temp_arr = np.abs(temp_arr)
        temp_arr = 20 * np.log10(temp_arr / temp_arr.max())
        temp_arr[temp_arr < my_decibel] = my_decibel
        return temp_arr
    else:
        raise TypeError


def give_f(points: int, sample_fre: float = 2*np.pi):
    """
    根据信号点数和采样频率给出绘制频谱时的横轴
    :param points:
    :param sample_fre:
    """
    if isinstance(points, int) and points > 0:
        fre_interval = sample_fre / points
        if is_even(points):
            return np.arange(-points//2 + 1, points//2 + 1) * fre_interval
        else:
            return np.arange(-(points - 1)//2, (points + 1)//2) * fre_interval
    else:
        raise ValueError
    pass


def fft_shift(arr: np.ndarray):
    length = arr.size
    if is_even(length):
        arr_temp = np.hstack((arr[length//2 + 1:], arr[0: length//2 + 1]))
    else:
        arr_temp = np.hstack((arr[(length+1)//2:], arr[0: (length+1)//2]))
    return arr_temp


if __name__ == '__main__':
    pass
