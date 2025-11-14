import random
import math
import torch
import numpy as np
from numpy.random import standard_normal, uniform

# 加噪
def awgn_np(data, snr_range):
    # 获取数据包的数量
    pkt_num = data.shape[0]
    # 在给定的信噪比范围内随机生成信噪比值
    SNRdB = uniform(snr_range[0], snr_range[-1], pkt_num)
    # 遍历每个数据包
    for pktIdx in range(pkt_num):
        # 获取当前数据包
        s = data[pktIdx]
        # 将信噪比值转换为线性比例
        SNR_linear = 10 ** (SNRdB[pktIdx] / 10)
        # 计算信号功率
        P = np.sum(np.abs(s) ** 2) / len(s)
        # 计算噪声功率
        N0 = P / SNR_linear
        # 生成复数高斯白噪声
        n = np.sqrt(N0 / 2) * (standard_normal(len(s)) + 1j * standard_normal(len(s)))
        # 将噪声添加到原始信号上
        data[pktIdx] = s + n
    # 返回添加噪声后的数据
    return data


# 幅值相位
def signal_transform_np(data, Euler=False):
    # 计算信号的幅度（振幅）
    am = np.abs(data)

    # 计算信号的相位
    ph = np.angle(data)

    # 如果使用欧拉表示法
    if Euler:
        # 计算实部和虚部的比值
        phc = data[:, 0:1, :, :] / am[:, np.newaxis, :, :]
        phs = data[:, 1:2, :, :] / am[:, np.newaxis, :, :]
        # 将幅度、实部比值和虚部比值拼接在一起
        data = np.concatenate((am[:, np.newaxis, :, :], phc, phs), axis=1)
    else:
        # 将幅度和相位拼接在一起
        data = np.concatenate((am[:, np.newaxis, :], ph[:, np.newaxis, :]), axis=1)

    return data


def augmentation_np(data, patch=False, ratio=True, overturn=True, flip=True, adp=1):
    # 如果需要对数据进行分片处理
    # if patch:
        # 如果分片长度不为0
        # if patch_len != 0:
        #     # 计算分片长度
        #     patch_len = int(signal_len - adp * patch_len)
        #
        #     # 随机生成一个起始点
        #     dt = random.randint(0, signal_len - patch_len)
        #
        #     # 对数据进行分片处理
        #     data = data[:, dt:dt + patch_len]

    # 如果需要进行比例变换
    if ratio:
        # 随机生成一个旋转角度
        r = (random.random() * math.pi * 2 - math.pi) * adp
        # 对数据的实部进行旋转变换
        datar0 = (math.cos(r) * data[:, 0, :, :] + math.sin(r) * data[:, 1, :, :])
        # 更新旋转角度
        r = r + math.pi / 2
        # 对数据的虚部进行旋转变换
        datar1 = (math.cos(r) * data[:, 0, :, :] + math.sin(r) * data[:, 1, :, :])

        # 将旋转后的数据拼接起来
        data = np.concatenate((datar0[:, np.newaxis, :, :], datar1[:, np.newaxis, :, :]), axis=1)

    # 如果需要进行翻转处理
    if overturn:
        # 随机决定是否翻转数据的第二个维度
        if random.getrandbits(1):
            data[:, 1, :, :] = -data[:, 1, :, :]

    # 如果需要进行翻转处理
    if flip:
        # 随机决定是否翻转数据的第一个维度
        if random.getrandbits(1):
            data = np.flip(data, axis=(1, 2))

    # 返回处理后的数据
    return data


def signal_transform_tensor(data, Euler=False):
    # 计算信号的幅度（振幅）
    am = (data[:, 0, :, :] ** 2 + data[:, 1, :, :] ** 2) ** 0.5
    # 计算信号的相位
    ph = torch.arctan(data[:, 0, :, :] / data[:, 1, :, :])
    
    # 如果使用欧拉表示法
    if Euler:
        # 计算实部和虚部的比值
        phc = data[:, 0:1,:,:] / am
        phs = data[:, 1:2,:,:] / am
        # 将幅度、实部比值和虚部比值拼接在一起
        data = torch.cat((am, phc, phs), dim=1)
    else:
        # 将幅度和相位拼接在一起
        data = torch.cat((am.unsqueeze(1), ph.unsqueeze(1)), dim=1)

    return data


def augmentation_tensor(data, patch=False, ratio=True, overturn=True, flip=True, adp=1):
    # 如果需要对数据进行分片处理
    # if patch:
        # 如果分片长度不为0
        # if patch_len != 0:
        #     # 计算分片长度
        #     patch_len = int(signal_len - adp * patch_len)
        #
        #     # 随机生成一个起始点
        #     dt = random.randint(0, signal_len - patch_len)
        #
        #     # 对数据进行分片处理
        #     data = data[:, dt:dt + patch_len]

    # 如果需要进行比例变换
    if ratio:
        # 随机生成一个旋转角度
        r = (random.random() * math.pi * 2 - math.pi) * adp
        # 对数据的实部进行旋转变换
        datar0 = (math.cos(r) * data[:, 0, :, :] + math.sin(r) * data[:, 1, :, :])
        # 更新旋转角度
        r = r + math.pi / 2
        # 对数据的虚部进行旋转变换
        datar1 = (math.cos(r) * data[:, 0, :, :] + math.sin(r) * data[:, 1, :, :])

        # 将旋转后的数据拼接起来
        data = torch.cat((datar0.unsqueeze(1), datar1.unsqueeze(1)), dim=1)

    # 如果需要进行翻转处理
    if overturn:
        # 随机决定是否翻转数据的第二个维度
        if random.getrandbits(1):
            data[:, 1, :, :] = -data[:, 1, :, :]

    # 如果需要进行翻转处理
    if flip:
        # 随机决定是否翻转数据的第一个维度
        if random.getrandbits(1):
            data = torch.flip(data, dims=[1, 2])

    # 返回处理后的数据
    return data

