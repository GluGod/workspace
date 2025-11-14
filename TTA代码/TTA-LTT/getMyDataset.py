import pickle
from scipy import signal
from scipy.signal import stft, hilbert, spectrogram, firwin
#from vmdpy import VMD
# from scipy import signal
#from skimage import filters, io
# from wlan_utils import my_timer_figure
# from pyhht import EMD
from data_augmentation import *

import torch
#import pywt
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.utils.data as Data
import matplotlib.pyplot as plt
from scipy.signal import stft
from utils import *
import seaborn as sns
#from PyEMD import EMD
# from PyEMD import EMD
# from vmdpy import VMD
# from scipy import signal
# from skimage import filters, io
# from pyhht import EMD
# from torchvision import transforms
# from pyhht.visualization import plot_imfs

# set_seed(22)


def Meta_Dataset():
    train_pkt_data = []
    val_pkt_data = []
    test_pkt_data = []
    data_root = "/home/data1/WZM/Dataset/temperature//"
    date = ["132", "2024_04_25-26", "2024_05_06-07", "2024_05_17-20", "2024_06_11-12", "2024_06_13"]
    # time = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    time = [2, 3, 4, 5]
    as_test = 1
    train_path = [
        data_root + date[time[0]] + "/train_data.pkt",
        data_root + date[time[1]] + "/train_data.pkt",
        data_root + date[time[2]] + "/train_data.pkt",
        data_root + date[time[3]] + "/train_data.pkt",
        # data_root + date[time[4]] + "/train_data.pkt"
    ]
    for i in train_path:
        data, label = data_read(i)
        known_dataset = Data.TensorDataset(data, label)
        train_pkt_data.append(known_dataset)

    train_loaders = [DataLoader(
        dataset=env,
        batch_size=64,
        num_workers=16,
        shuffle=True)
        for env in train_pkt_data]

    test_path = [
        data_root + date[time[0]] + "/test_data.pkt",
        data_root + date[time[1]] + "/test_data.pkt",
        data_root + date[time[2]] + "/test_data.pkt",
        data_root + date[time[3]] + "/test_data.pkt",
        # data_root + date[time[4]] + "/test_data.pkt"
    ]
    for i in test_path:
        data, label = data_read(i)
        known_dataset = Data.TensorDataset(data, label)
        val_pkt_data.append(known_dataset)

    val_loaders = [DataLoader(
        dataset=env,
        batch_size=64,
        num_workers=16,
        shuffle=True)
        for env in val_pkt_data]


    data_path = [
        data_root + date[as_test] + "/train_data.pkt",
        data_root + date[as_test] + "/test_data.pkt"
    ]
    for i in data_path:
        data, label = data_read(i)
        known_dataset = Data.TensorDataset(data, label)
        test_pkt_data.append(known_dataset)

    test_loaders = [DataLoader(
        dataset=env,
        batch_size=64,
        num_workers=16,
        shuffle=True)
        for env in test_pkt_data]
    return train_loaders, val_loaders, test_loaders


class myDataset(Dataset):
    def __init__(self, data_path, flag=None):
        # pkt_data = []
        # if flag=="1":
        #     data_root = "/home/data1/WZM/Dataset/temperature//"
        #     # pkt_data = []
        #     data_path=[
        #         data_root + "2024_04_25-26/train_data.pkt",
        #         data_root + "2024_04_28-30/train_data.pkt",
        #         data_root + "2024_05_06-07/train_data.pkt",
        #         data_root + "2024_05_08-09/train_data.pkt"
        #     ]
        #     for i in range(len(data_path)):
        #         with open(data_path[i], "rb") as f:
        #             pkt_data.extend(pickle.load(f))
        #
        # elif flag=="2":
        #     data_root = "/home/data1/WZM/Dataset/temperature//"
        #     data_path=[
        #         data_root + "2024_04_25-26/test_data.pkt",
        #         data_root + "2024_04_28-30/test_data.pkt",
        #         data_root + "2024_05_06-07/test_data.pkt",
        #         data_root + "2024_05_08-09/test_data.pkt"
        #     ]
        #     for i in range(len(data_path)):
        #         with open(data_path[i], "rb") as f:
        #             pkt_data.extend(pickle.load(f))
        # else:
        #     data_root = "/home/data1/WZM/Dataset/temperature//"
        #     data_path = [
        #         data_root + "2024_05_17-20/train_data.pkt",
        #         data_root + "2024_05_17-20/test_data.pkt"
        #     ]
        #     for i in range(len(data_path)):
        #         with open(data_path[i], "rb") as f:
        #             pkt_data.extend(pickle.load(f))
        #
        # if flag == "1":
        with open(data_path, "rb") as f:
            pkt_data = pickle.load(f)
            # data_np = np.zeros((len(pkt_data), 320), dtype=complex)
            # label = np.zeros(len(pkt_data), dtype=int)
            data_np_tr = []
            label_tr =[]
            data_np_te=[]
            label_te=[]
            temperature = [[] for _ in range(10)]
            tem = [
                [33, 39],
                [36, 41],
                [39, 43],
                [37, 42],
                [33, 38],
                [40, 42],
                [36, 39],
                [41, 43],
                [35, 39],
                [37, 39]
            ]
            # app = np.zeros(len(pkt_data), dtype=str)
            for i, data in enumerate(pkt_data):
                for n in range(10):
                    if int(data.dev_label) == n:
                        # data_i = data.data[100:100 + 320]
                        # data_np[i] = data_i
                        # label[i] = data.dev_label
                        temperature[n].append(float(data.temperature))
            for i, data in enumerate(pkt_data):
                for n in range(10):
                    if int(data.dev_label) == n:
                        if float(data.temperature) == tem[n][0]:
                            data_i = data.data[100:100 + 320]
                            # cfo_i = data.cfo
                            # data_i = freqCompensation(data_i, float(cfo_i))
                            data_np_tr.append(data_i)
                            label_tr.append(data.dev_label)
                        if float(data.temperature) == tem[n][1]:
                            data_i = data.data[100:100 + 320]
                            # cfo_i = data.cfo
                            # data_i = freqCompensation(data_i, float(cfo_i))
                            data_np_te.append(data_i)
                            label_te.append(data.dev_label)
        if flag =="1":          # 训练集温度
            data_np = data_norm(np.array(data_np_tr))
            label = np.array(label_tr).flatten()
        elif flag == "2":       # 相同温度
            data_np = data_norm(np.array(data_np_tr))
            label = np.array(label_tr).flatten()
        else:                   # 不同温度
            # print("error")
            data_np = data_norm(np.array(data_np_te))
            label = np.array(label_te).flatten()
            # temp = np.unique(temperature)
        # liat ,count1=np.unique(np.array(label_tr), return_counts=True)
        # list, count2 =np.unique(np.array(label_te), return_counts=True)
        data_np = awgn_np(data_np, snr_range=[0, 20])
        # my_timer_figure(data_np[0])
        data_np = data_norm(data_np)

        # raw_data_tensor = torch.from_numpy(data_np).unsqueeze(2)
        # self.raw_data = torch.cat((raw_data_tensor.real.unsqueeze(1).float(), raw_data_tensor.imag.unsqueeze(1).float()), dim=1)
        #
        # data_fft1 = np.abs(np.fft.fft(data_np))
        # fft_result = np.fft.fft(data_np)
        # psd_data = np.zeros_like(data_np)
        # # 计算每行数据的功率谱密度
        # for i in range(data_np.shape[0]):
        #     psd_data[i] = np.abs(fft_result[i]) ** 2 / np.linalg.norm(fft_result[i])
        # fft_data_tensor = torch.from_numpy(psd_data).unsqueeze(2)
        # self.fft_data = torch.cat((fft_data_tensor.real.unsqueeze(1).float(), fft_data_tensor.imag.unsqueeze(1).float()), dim=1)

        # frequencies, times, data_stft = stft(data_np, nperseg=128, noverlap=115, return_onesided=False)
        stft_data_tensor = torch.from_numpy(data_np).unsqueeze(2)
        # stft_data_tensor = torch.from_numpy(data_stft)

        # self.stft_data = torch.cat((data_tensor.real.unsqueeze(1).float(), data_tensor.imag.unsqueeze(1).float()), dim=1)
        self.stft_data = torch.cat((stft_data_tensor.real.unsqueeze(1).float(), stft_data_tensor.imag.unsqueeze(1).float()), dim=1)

        self.label = torch.LongTensor(label)
        # print("123")

    def __getitem__(self, index):
        # raw_data = self.raw_data[index]
        # fft_data = self.fft_data[index]
        stft_data = self.stft_data[index]
        label = self.label[index]
        return stft_data, label

    def __len__(self):
        return len(self.label)

# 单天相同不同温度数据
class myDataset1(Dataset):
    def __init__(self, data_path, flag=None):
        # pkt_data = []
        # if flag=="1":
        #     data_root = "/home/data1/WZM/Dataset/temperature//"
        #     # pkt_data = []
        #     data_path=[
        #         data_root + "2024_04_25-26/train_data.pkt",
        #         data_root + "2024_04_28-30/train_data.pkt",
        #         data_root + "2024_05_06-07/train_data.pkt",
        #         data_root + "2024_05_08-09/train_data.pkt"
        #     ]
        #     for i in range(len(data_path)):
        #         with open(data_path[i], "rb") as f:
        #             pkt_data.extend(pickle.load(f))
        #
        # elif flag=="2":
        #     data_root = "/home/data1/WZM/Dataset/temperature//"
        #     data_path=[
        #         data_root + "2024_04_25-26/test_data.pkt",
        #         data_root + "2024_04_28-30/test_data.pkt",
        #         data_root + "2024_05_06-07/test_data.pkt",
        #         data_root + "2024_05_08-09/test_data.pkt"
        #     ]
        #     for i in range(len(data_path)):
        #         with open(data_path[i], "rb") as f:
        #             pkt_data.extend(pickle.load(f))
        # else:
        #     data_root = "/home/data1/WZM/Dataset/temperature//"
        #     data_path = [
        #         data_root + "2024_05_17-20/train_data.pkt",
        #         data_root + "2024_05_17-20/test_data.pkt"
        #     ]
        #     for i in range(len(data_path)):
        #         with open(data_path[i], "rb") as f:
        #             pkt_data.extend(pickle.load(f))
        #
        # if flag == "1":
        with open(data_path, "rb") as f:
            pkt_data = pickle.load(f)
            # data_np = np.zeros((len(pkt_data), 320), dtype=complex)
            # label = np.zeros(len(pkt_data), dtype=int)
            data_np_tr = []
            label_tr =[]
            data_np_te=[]
            label_te=[]
            temperature = [[] for _ in range(10)]
            tem = [
                [33, 39],
                [36, 41],
                [39, 43],
                [37, 42],
                [33, 38],
                [40, 42],
                [36, 39],
                [41, 43],
                [35, 39],
                [37, 39]
            ]
            # app = np.zeros(len(pkt_data), dtype=str)
            for i, data in enumerate(pkt_data):
                for n in range(10):
                    if int(data.dev_label) == n:
                        # data_i = data.data[100:100 + 320]
                        # data_np[i] = data_i
                        # label[i] = data.dev_label
                        temperature[n].append(float(data.temperature))
            for i, data in enumerate(pkt_data):
                # for n in range(10):
                if int(data.app_label) == 0:
                    # if float(data.temperature) == tem[n][0]:
                    data_i = data.data[100:100 + 320]
                    # cfo_i = data.cfo
                    # data_i = freqCompensation(data_i, float(cfo_i))
                    data_np_tr.append(data_i)
                    label_tr.append(data.dev_label)
                    # if float(data.temperature) == tem[n][1]:
                if int(data.app_label) == 3:
                    data_i = data.data[100:100 + 320]
                    # cfo_i = data.cfo
                    # data_i = freqCompensation(data_i, float(cfo_i))
                    data_np_te.append(data_i)
                    label_te.append(data.dev_label)
        if flag =="1":          # 训练集温度
            data_np = np.array(data_np_tr)
            label = np.array(label_tr).flatten()
        elif flag == "2":       # 相同温度
            data_np = np.array(data_np_tr)
            label = np.array(label_tr).flatten()
        else:                   # 不同温度
            # print("error")
            data_np = np.array(data_np_te)
            label = np.array(label_te).flatten()
            # temp = np.unique(temperature)
        # liat ,count1=np.unique(np.array(label_tr), return_counts=True)
        # list, count2 =np.unique(np.array(label_te), return_counts=True)
        # data_np = awgn_np(data_np, snr_range=[0, 20])
        # my_timer_figure(data_np[0])
        data_np = data_norm(data_np)

        # raw_data_tensor = torch.from_numpy(data_np).unsqueeze(2)
        # self.raw_data = torch.cat((raw_data_tensor.real.unsqueeze(1).float(), raw_data_tensor.imag.unsqueeze(1).float()), dim=1)
        #
        # data_fft1 = np.abs(np.fft.fft(data_np))
        # fft_result = np.fft.fft(data_np)
        # psd_data = np.zeros_like(data_np)
        # # 计算每行数据的功率谱密度
        # for i in range(data_np.shape[0]):
        #     psd_data[i] = np.abs(fft_result[i]) ** 2 / np.linalg.norm(fft_result[i])
        # fft_data_tensor = torch.from_numpy(psd_data).unsqueeze(2)
        # self.fft_data = torch.cat((fft_data_tensor.real.unsqueeze(1).float(), fft_data_tensor.imag.unsqueeze(1).float()), dim=1)

        # frequencies, times, data_stft = stft(data_np, nperseg=128, noverlap=115, return_onesided=False)
        stft_data_tensor = torch.from_numpy(data_np).unsqueeze(2)
        # stft_data_tensor = torch.from_numpy(data_stft)

        # self.stft_data = torch.cat((data_tensor.real.unsqueeze(1).float(), data_tensor.imag.unsqueeze(1).float()), dim=1)
        self.stft_data = torch.cat((stft_data_tensor.real.unsqueeze(1).float(), stft_data_tensor.imag.unsqueeze(1).float()), dim=1)

        self.label = torch.LongTensor(label)
        # print("123")

    def __getitem__(self, index):
        # raw_data = self.raw_data[index]
        # fft_data = self.fft_data[index]
        stft_data = self.stft_data[index]
        label = self.label[index]
        return stft_data, label

    def __len__(self):
        return len(self.label)


class myDataset2(Dataset):
    def __init__(self, data_path, flag=None):
        pkt_data = []
        data_root = "/home/data1/WZM/Dataset/temperature//"
        if flag=="1":
            data_path=[
                data_root + "2024_04_25-26/train_data.pkt",
                data_root + "2024_04_28-30/train_data.pkt",
                data_root + "2024_05_06-07/train_data.pkt",
                data_root + "2024_05_08-09/train_data.pkt"
            ]
            for i in range(len(data_path)):
                with open(data_path[i], "rb") as f:
                    pkt_data.extend(pickle.load(f))
        elif flag=="2":
            data_path=[
                data_root + "2024_04_25-26/test_data.pkt",
                data_root + "2024_04_28-30/test_data.pkt",
                data_root + "2024_05_06-07/test_data.pkt",
                data_root + "2024_05_08-09/test_data.pkt"
            ]
            for i in range(len(data_path)):
                with open(data_path[i], "rb") as f:
                    pkt_data.extend(pickle.load(f))
        else:
            data_path = [
                data_root + "2024_05_17-20/train_data.pkt",
                data_root + "2024_05_17-20/test_data.pkt"
            ]
            for i in range(len(data_path)):
                with open(data_path[i], "rb") as f:
                    pkt_data.extend(pickle.load(f))
        pkt_data = np.array(pkt_data)
        # idx1 = [8, 12, 16, 20, 24, 28, 36, 40, 44, 48, 52, 56]
        data_np = np.zeros((len(pkt_data), 320), dtype=complex)
        cfo = np.zeros(len(pkt_data), dtype=float)
        label = np.zeros(len(pkt_data), dtype=int)
        for i, data in enumerate(pkt_data):
            data_i = data.data[100:100 + 320]
            a = float(data.cfo)
            # cfo[i] = float(data.cfo)
            # my_timer_figure(data_i)
            # cfo_coarse = coarseCfoEstimate(data_i[:160])
            # data_i = freqCompensation(preamble=data_i, coarseFreqOffset=cfo_coarse)
            # cfo_fine = fineCfoEstimate(data_i[160:])
            # data_i = freqCompensation(preamble=data_i, coarseFreqOffset=cfo_fine)
            # data_i = phaseCompensation(data_i)
            # my_timer_figure(data_i)

            # data_j = data_i
            # cfo_m = coarseCfoEstimate(data_j[:160])
            # data_j = freqCompensation(preamble=data_j, coarseFreqOffset=cfo_m)
            # cfo_n = fineCfoEstimate(data_j[160:])

            cfo[i] = a
            # my_timer_figure(data_i)
            # data_i = freqCompensation(data_i, cfo_i)
            # cfo_j = freqCfoEstimate(data_i)
            # my_timer_figure(data_i)
            data_np[i] = data_i
            label[i] = data.dev_label
        # data_np = data_norm(data_np)
        # sts1 = data_np[:, 16:16 * 5]
        # sts2 = data_np[:, 16 * 5:16 * 9]
        # lts1 = data_np[:, 192:192 + 64]
        # lts2 = data_np[:, 192 + 64:192 + 64 * 2]
        #
        # rff1 = np.log(np.fft.fft(sts1)) - np.log(np.fft.fft(lts1))
        # rff2 = np.log(np.fft.fft(sts2)) - np.log(np.fft.fft(lts2))
        # rff = np.concatenate((rff1, rff2), axis=-1)
        # rff1 = min_max_normalize(rff1)
        # rff2 = min_max_normalize(rff2)
        # rff = data_norm(rff)
        # rff2 = data_norm(rff2)
        #
        # # 计算平均cfo
        # # # 频偏分布曲线图
        # # # 设置Seaborn的样式
        # # sns.set(style="whitegrid")
        # # # 创建一个新的绘图对象
        # # # plt.figure(figsize=(10, 6))
        # # colors = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple', 'gray', 'black', 'brown', 'pink']
        # # self.cfo_list = []
        # # for i in range(10):
        # #     idx = np.where(label == i)
        # #     cfo_i = cfo[idx]
        # #     # 使用Seaborn绘制概率分布曲线图
        # #     # sns.kdeplot(cfo_i, shade=True, color=colors[i], bw_adjust=0.5, label="class"+str(i))
        # #     delta_f = (np.max(cfo_i)+np.min(cfo_i))/2
        # #     lambda_f = (np.max(cfo_i)-np.min(cfo_i))/2
        # #     cfo_i_database = np.concatenate((np.max(cfo_i), np.min(cfo_i), delta_f, lambda_f), axis=None)
        # #     self.cfo_list.append(cfo_i_database)
        # # # plt.legend()
        # # # plt.show()
        #
        #
        # # frequencies, times, data_stft = stft(data_np, nperseg=128, noverlap=115, return_onesided=False)
        # # stft_data_tensor = torch.from_numpy(rff).unsqueeze(2)
        # data_np = data_norm(data_np)
        stft_data_tensor = torch.from_numpy(data_np)
        # stft_data_tensor2 = torch.from_numpy(rff2).unsqueeze(2)
        self.cfo = cfo
        # self.stft_data = torch.cat((stft_data_tensor1.real.unsqueeze(1).float(), stft_data_tensor1.imag.unsqueeze(1).float(), stft_data_tensor2.real.unsqueeze(1).float(), stft_data_tensor2.imag.unsqueeze(1).float()), dim=1)
        # stft_data_tensor = torch.from_numpy(rff1).unsqueeze(2).unsqueeze(1).float()
        # stft_data_tensor = torch.from_numpy(data_stft)
        # self.stft_data = torch.cat((stft_data_tensor.real.unsqueeze(1).float(), stft_data_tensor.imag.unsqueeze(1).float()), dim=1)
        self.stft_data = stft_data_tensor
        self.label = torch.LongTensor(label)
        self.label_np = label

    def __getitem__(self, index):
        # raw_data = self.raw_data[index]
        # fft_data = self.fft_data[index]
        stft_data = self.stft_data[index]
        label = self.label[index]
        cfo = self.cfo[index]
        return stft_data, label

    def __len__(self):
        return len(self.label)


class myDataset3(Dataset):
    def __init__(self, data_path, flag=None, delta_cfo=None):
        pkt_data = []
        data_root = "/home/data1/WZM/Dataset/temperature//"
        if flag=="1":
            data_path=[
                data_root + "2024_04_25-26/train_data.pkt",
                data_root + "2024_04_28-30/train_data.pkt",
                data_root + "2024_05_06-07/train_data.pkt",
                data_root + "2024_05_08-09/train_data.pkt"
            ]
            for i in range(len(data_path)):
                with open(data_path[i], "rb") as f:
                    pkt_data.extend(pickle.load(f))
        elif flag=="2":
            data_path=[
                data_root + "2024_04_25-26/test_data.pkt",
                data_root + "2024_04_28-30/test_data.pkt",
                data_root + "2024_05_06-07/test_data.pkt",
                data_root + "2024_05_08-09/test_data.pkt"
            ]
            for i in range(len(data_path)):
                with open(data_path[i], "rb") as f:
                    pkt_data.extend(pickle.load(f))
        else:
            data_path = [
                data_root + "2024_05_17-20/train_data.pkt",
                data_root + "2024_05_17-20/test_data.pkt"
            ]
            for i in range(len(data_path)):
                with open(data_path[i], "rb") as f:
                    pkt_data.extend(pickle.load(f))
        pkt_data = np.array(pkt_data)
        # idx1 = [8, 12, 16, 20, 24, 28, 36, 40, 44, 48, 52, 56]
        data_np = np.zeros((len(pkt_data), 320), dtype=complex)
        cfo = np.zeros(len(pkt_data), dtype=float)
        label = np.zeros(len(pkt_data), dtype=int)
        for i, data in enumerate(pkt_data):
            data_i = data.data[100:100 + 320]
            cfo[i] = float(data.cfo)
            # my_timer_figure(data_i)
            data_i = freqCompensation(data_i, delta_cfo[data.dev_label])
            # my_timer_figure(data_i)
            data_np[i] = data_i
            label[i] = data.dev_label
        data_np = data_norm(data_np)
        sts1 = data_np[:, 16:16 * 5]
        sts2 = data_np[:, 16 * 5:16 * 9]
        lts1 = data_np[:, 192:192 + 64]
        lts2 = data_np[:, 192 + 64:192 + 64 * 2]
        # rff = np.concatenate((sts1, sts2, lts1, lts2), axis=-1)

        rff1 = np.log(np.fft.fft(sts1)) - np.log(np.fft.fft(lts1))
        rff2 = np.log(np.fft.fft(sts2)) - np.log(np.fft.fft(lts2))

        # 计算平均cfo
        self.cfo_list = []
        for i in range(10):
            idx = np.where(label == i)
            cfo_i = cfo[idx]
            cfo_i_mean = np.mean(cfo_i)
            self.cfo_list.append(cfo_i_mean)

        # frequencies, times, data_stft = stft(data_np, nperseg=128, noverlap=115, return_onesided=False)
        # stft_data_tensor = torch.from_numpy(rff).unsqueeze(2)
        stft_data_tensor1 = torch.from_numpy(rff1).unsqueeze(2)
        stft_data_tensor2 = torch.from_numpy(rff2).unsqueeze(2)
        self.cfo = cfo
        self.stft_data = torch.cat((stft_data_tensor1.real.unsqueeze(1).float(), stft_data_tensor1.imag.unsqueeze(1).float(), stft_data_tensor2.real.unsqueeze(1).float(), stft_data_tensor2.imag.unsqueeze(1).float()), dim=1)
        # stft_data_tensor = torch.from_numpy(rff1).unsqueeze(2).unsqueeze(1).float()
        # stft_data_tensor = torch.from_numpy(data_stft)
        # self.stft_data = torch.cat((stft_data_tensor.real.unsqueeze(1).float(), stft_data_tensor.imag.unsqueeze(1).float()), dim=1)
        # self.stft_data = stft_data_tensor
        self.label = torch.LongTensor(label)


    def __getitem__(self, index):
        # raw_data = self.raw_data[index]
        # fft_data = self.fft_data[index]
        stft_data = self.stft_data[index]
        label = self.label[index]
        cfo = self.cfo[index]
        return stft_data, label, cfo

    def __len__(self):
        return len(self.label)

def plot_IQimage(data):
    # 创建时间轴
    time = np.arange(0, len(data))
    # 创建图形
    plt.figure(figsize=(15, 10))

    # 1. 绘制I路信号（实部）
    plt.subplot(3, 1, 1)
    plt.plot(time, np.real(data), 'b', label='I (Real)')
    plt.title('I path signal (Real)')
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()

    # 2. 绘制Q路信号（虚部）
    plt.subplot(3, 1, 2)
    plt.plot(time, np.imag(data), 'r', label='Q (Imaginary)')
    plt.title('Q path signal (Imaginary)')
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()

    # 3. 绘制星座图（IQ平面）
    plt.subplot(3, 1, 3)
    plt.scatter(np.real(data), np.imag(data), c=time, cmap='viridis', alpha=0.7)
    plt.title('Constellation diagram (IQ plane)')
    plt.xlabel('I (Real)')
    plt.ylabel('Q (Imaginar)')
    plt.grid(True)
    plt.colorbar(label='Time (samples)')
    plt.axis('equal')  # 确保x和y轴比例相同
    # 添加标题
    plt.suptitle('IQ signal analysis', fontsize=16)
    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为总标题留出空间
    # 显示图形
    plt.show()

# 单天数据做训练集
class myDataset4(Dataset):
    def __init__(self, data_path, flag=None):
        with open(data_path, "rb") as f:
            pkt_data = pickle.load(f)
        pkt_data = np.array(pkt_data)
        idx = [4, 8, 12, 16, 20, 24, 40, 44, 48, 52, 56, 60]
        data_np = np.zeros((len(pkt_data), 320), dtype=complex)
        cfo = np.zeros(len(pkt_data), dtype=float)
        label = np.zeros(len(pkt_data), dtype=int)
        for i, data in enumerate(pkt_data):
            data_i = data.data[100:100 + 320]
            a = float(data.cfo)
            # plot_IQimage(data_i)

            #delta_f = freqCfoEstimate(data_i)
            # 粗估计
            cfo_coarse = coarseCfoEstimate(data_i[:160])
            data_i = freqCompensation(preamble=data_i, coarseFreqOffset=cfo_coarse)
            # plot_IQimage(data_i)
            # 细估计
            cfo_fine = fineCfoEstimate(data_i[160:])
            data_i = freqCompensation(preamble=data_i, coarseFreqOffset=cfo_fine)
            # plot_IQimage(data_i)

            data_i = phaseCompensation(data_i)
            # plot_IQimage(data_i)

            # my_timer_figure(data_i)
            # data_j = data_i
            # cfo_m = coarseCfoEstimate(data_j[:160])
            # data_j = freqCompensation(preamble=data_j, coarseFreqOffset=cfo_m)
            # cfo_n = fineCfoEstimate(data_j[160:])

            cfo[i] = a
            # my_timer_figure(data_i)
            # data_i = freqCompensation(data_i, a)
            # data_i = phaseCompensation(data_i)
            # my_timer_figure(data_i)
            # cfo_j = freqCfoEstimate(data_i)
            # my_timer_figure(data_i)
            data_np[i] = data_i
            label[i] = data.dev_label

        data_np = data_norm_np(data_np)
        # IQ
        self.stft_data = torch.from_numpy(data_np)
        #self.stft_data = torch.cat((stft_data_tensor.real.unsqueeze(1).float(), stft_data_tensor.imag.unsqueeze(1).float()), dim=1)

        # FFT
        # data_np = np.fft.fft(data_np)
        # stft_data_tensor = torch.from_numpy(data_np)
        # self.stft_data = torch.cat((stft_data_tensor.real.unsqueeze(1).float(), stft_data_tensor.imag.unsqueeze(1).float()), dim=1)

        # PSD
        # data_np = data_norm_np(data_np)
        # # 对每行数据进行FFT计算
        # fft_result = np.fft.fft(data_np)
        # psd_data = np.zeros_like(data_np)
        # # 计算每行数据的功率谱密度
        # for i in range(data_np.shape[0]):
        #     psd_data[i] = np.abs(fft_result[i]) ** 2 / np.linalg.norm(fft_result[i])
        # stft_data_tensor = torch.from_numpy(psd_data)
        # self.stft_data = stft_data_tensor.unsqueeze(1).float()      # (10000, 1, 256)

        # STFT
        # frequencies, times, data_stft = stft(data_np, nperseg=128, noverlap=115, return_onesided=False)
        # stft_data_tensor = torch.from_numpy(data_stft)
        # self.stft_data = torch.cat((stft_data_tensor.real.unsqueeze(1).float(), stft_data_tensor.imag.unsqueeze(1).float()), dim=1)

        # ChiHis
        # if flag == "1":
        #     data_np = awgn(data_np, range(20, 80))
        # ChannelIndSpectrogramObj = ChannelIndSpectrogram()
        # data_np = ChannelIndSpectrogramObj.channel_ind_spectrogram(data_np)
        # stft_data_tensor = torch.from_numpy(data_np)
        # self.stft_data = stft_data_tensor.unsqueeze(1).float()

        # Dolos
        # sts1 = data_np[:, 16:16 * 5]
        # sts2 = data_np[:, 16 * 5:16 * 9]
        # lts1 = data_np[:, 192:192 + 64]
        # lts2 = data_np[:, 192 + 64:192 + 64 * 2]
        # sts = (sts1+sts2)/2
        # lts = (lts1+lts2)/2
        # rff = np.log(np.abs(np.fft.fft(sts))) - np.log(np.abs(np.fft.fft(lts)))
        # rff = rff[:, idx]
        # stft_data_tensor = torch.from_numpy(rff)
        # self.stft_data = stft_data_tensor.unsqueeze(1).float()

        # CWT
        # data_xiaobo = np.empty((data_np.shape[0], 32, 320), dtype=complex)
        # wavelet = 'gaus1'  # 选择小波函数，这里使用 Morlet 小波
        # for i in range(data_np.shape[0]):
        # # 进行小波变换
        #     coeffs, freqs = pywt.cwt(data_np[i], scales=np.arange(1, 32+1), wavelet=wavelet)
        #     data_xiaobo[i] = coeffs
        # stft_data_tensor = torch.from_numpy(data_xiaobo)
        # self.stft_data = torch.cat((stft_data_tensor.real.unsqueeze(1).float(), stft_data_tensor.imag.unsqueeze(1).float()), dim=1)

        # HHT
        # emd = EMD()
        # t = np.arange(320)
        # # imfs = emd(data_np)
        # data_np_real = np.real(data_np)
        # data_np_imag = np.imag(data_np)
        # hht_real = np.empty((data_np.shape[0], 5, 320))
        # hht_imag = np.empty((data_np.shape[0], 5, 320))
        # for i in range(data_np.shape[0]):
        #     imfs_real = emd(data_np_real[i], t)[:5]
        #     imfs_imag = emd(data_np_imag[i], t)[:5]
        #     hht_real[i] = hilbert(imfs_real)
        #     hht_imag[i] = hilbert(imfs_imag)
        # data_norm_real = torch.from_numpy(hht_real).float()
        # data_norm_imag = torch.from_numpy(hht_imag).float()
        # print("123")
        # self.stft_data = torch.cat((data_norm_real.unsqueeze(1), data_norm_imag.unsqueeze(1)), dim=1)   # (10000, 2, 5, 320)

        # EMD
        # emd = EMD()
        # t = np.arange(320)
        # # imfs = emd(data_np)
        # data_np_real = np.real(data_np)
        # data_np_imag = np.imag(data_np)
        # emd_real = np.empty((data_np.shape[0], 5, 320))
        # emd_imag = np.empty((data_np.shape[0], 5, 320))
        # for i in range(data_np.shape[0]):
        #     imfs_real = emd(data_np_real[i], t)
        #     imfs_imag = emd(data_np_imag[i], t)
        #     emd_real[i] = imfs_real
        #     emd_imag[i] = imfs_imag
        # data_norm_real = torch.from_numpy(emd_real).float()
        # data_norm_imag = torch.from_numpy(emd_imag).float()
        # print("123")
        # self.stft_data = torch.cat((data_norm_real.unsqueeze(1), data_norm_imag.unsqueeze(1)), dim=1)    # (10000, 2, 5, 320)

        # Gabor
        # data_np = data_np[:, :, np.newaxis]
        # # 定义 Gabor 参数
        # frequency = 0.2
        # theta = 0
        # sigma_x = None
        # sigma_y = None
        # n_stds = 3
        # # 创建一个空数组用于保存结果
        # gabor_result_real = np.zeros_like(data_np)
        # gabor_result_imag = np.zeros_like(data_np)
        # # 遍历每行数据并应用 Gabor 变换
        # for i in range(data_np.shape[0]):
        #     row_data = data_np[i]  # 获取每条数据
        #     filtered_row_real, _ = filters.gabor(np.real(row_data), frequency=frequency, theta=theta, sigma_x=sigma_x, sigma_y=sigma_y, n_stds=n_stds)  # 应用 Gabor 滤波器
        #     filtered_row_imag, _ = filters.gabor(np.real(row_data), frequency=frequency, theta=theta, sigma_x=sigma_x, sigma_y=sigma_y, n_stds=n_stds)  # 应用 Gabor 滤波器
        #     gabor_result_real[i] = filtered_row_real  # 保存结果到 gabor_result 中
        #     gabor_result_imag[i] = filtered_row_imag
        #     # my_timer_figure(gabor_result[i])
        # data_norm_real = torch.from_numpy(gabor_result_real).squeeze(2).float()
        # data_norm_imag = torch.from_numpy(gabor_result_imag).squeeze(2).float()
        # self.stft_data = torch.cat((data_norm_real.unsqueeze(1), data_norm_imag.unsqueeze(1)), dim=1)  # (10000, 2, 320)

        # VMD
        # data = np.zeros((data_np.shape[0], 5, 320))
        # alpha = 2000
        # tau = 0  # tau 噪声容限，即允许重构后的信号与原始信号有差别。
        # K = 5  # K 分解模态（IMF）个数
        # DC = 0  # DC 若为0则让第一个IMF为直流分量/趋势向量
        # init = 1  # init 指每个IMF的中心频率进行初始化。当初始化为1时，进行均匀初始化。
        # tol = 1e-7  # 控制误差大小常量，决定精度与迭代次数
        # for i in range(data_np.shape[0]):
        #     # 对信号进行VMD分解
        #     data[i], u_hat, omega = VMD(data_np[i], alpha, tau, K, DC, init, tol)
        # data_norm_real = torch.from_numpy(np.real(data)).float()
        # data_norm_imag = torch.from_numpy(np.imag(data)).float()
        # self.stft_data = torch.cat((data_norm_real.unsqueeze(1), data_norm_imag.unsqueeze(1)), dim=1)

        # EPS
        # data_np_real = np.zeros(data_np.shape)
        # data_np_imag = np.zeros(data_np.shape)
        # hilbert_filter = firwin(101, 0.5)
        # for i in range(data_np.shape[0]):
        #     envelope_I = np.abs(np.convolve(np.real(data_np[i]), hilbert_filter, mode='same'))
        #     envelope_Q = np.abs(np.convolve(np.imag(data_np[i]), hilbert_filter, mode='same'))
        #     envelope_I = np.real(data_np[i]) + envelope_I * 1j
        #     envelope_Q = np.imag(data_np[i]) + envelope_Q * 1j
        #     envelope_I_dc_removed = envelope_I - np.mean(envelope_I, axis=0)
        #     envelope_Q_dc_removed = envelope_Q - np.mean(envelope_Q, axis=0)
        #     # 计算功率谱密度（PSD）
        #     psd_I = np.abs(np.fft.fft(envelope_I_dc_removed, axis=0)) ** 2 / np.linalg.norm(np.fft.fft(envelope_I_dc_removed, axis=0))
        #     psd_Q = np.abs(np.fft.fft(envelope_Q_dc_removed, axis=0)) ** 2 / np.linalg.norm(np.fft.fft(envelope_Q_dc_removed, axis=0))
        #     # 归一化功率谱
        #     freq = np.fft.fftfreq(envelope_I_dc_removed.shape[0], d=1)  # 频率轴
        #     data_np_real[i] = psd_I / np.max(psd_I, axis=0)
        #     data_np_imag[i] = psd_Q / np.max(psd_Q, axis=0)
        # data_norm_real = torch.from_numpy(data_np_real).float()
        # data_norm_imag = torch.from_numpy(data_np_imag).float()
        # self.stft_data = torch.cat((data_norm_real.unsqueeze(1), data_norm_imag.unsqueeze(1)), dim=1)   # (10000, 2, 320)

        # CLPS
        # data_np = data_norm_np()
        # p_log = np.log(np.abs(np.fft.fft(data_np)))
        # clps = 2 * 20e6 * (p_log - np.mean(p_log))
        # clps = torch.from_numpy(clps)
        # self.stft_data = clps.unsqueeze(1).float()

        # Ours
        # data_np = data_norm_np(data_np)
        # stft_data_tensor = torch.from_numpy(data_np)
        # self.stft_data = stft_data_tensor

        self.label = torch.LongTensor(label)
        self.label_np = label

    def __getitem__(self, index):
        # raw_data = self.raw_data[index]
        # fft_data = self.fft_data[index]
        stft_data = self.stft_data[index]
        label = self.label[index]
        # cfo = self.cfo[index]
        return stft_data, label

    def __len__(self):
        return len(self.label)
def data_norm1(data_np):
    data_norm = np.zeros(data_np.shape, dtype=complex)
    for i in range(data_np.shape[0]):
        sig_amplitude = np.abs(data_np[i])
        rms = np.sqrt(np.mean(sig_amplitude ** 2))
        data_norm[i] = data_np[i] / rms
    return data_norm
def data_read(path):
    # 读取.pkt文件
    data = read_pkt_file(path)
    data_np = np.zeros((len(data), 320), dtype=complex)
    train_label = np.zeros(len(data), dtype=int)
    #K=ChannelIndSpectrogram()
    for i, data in enumerate(data):
        data_i = data.data[100:100 + 320]
        data_np[i] = data_i
        train_label[i] = data.dev_label
    # data = np.load(root + path)
    train_data = data_np
    train_data = data_norm1(train_data)
    #train_data = K.channel_ind_spectrogram(train_data)
    #local_data, shot, long = localSequenceGenerate()
    print(train_data.shape)
    #train_data = data_norm(train_data - np.concatenate((shot, long), axis=0))

    # train_data_real, train_data_imag = np.real(train_data).reshape(len(train_data), 1,320), np.imag(
    #     train_data).reshape(len(train_data), 1, 320)
    # train_data = np.concatenate((train_data_real, train_data_imag), axis=1)
    train_label = torch.LongTensor(train_label)
    train_data = torch.from_numpy(train_data)
    return train_data, train_label


def read_pkt_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


# 多天数据合并做训练集
class myDataset5(Dataset):
    def __init__(self, data_path=None, flag=None):
        train_pkt_data = []
        val_pkt_data = []
        test_pkt_data = []
        data_root = "/home/data1/WZM/Dataset/temperature//"
        #date = ["132", "2024_04_25-26", "2024_04_28-30", "2024_05_06-07", "2024_05_08-09", "2024_05_17-20", "2024_06_03-04", "2024_06_05-06", "2024_06_11-12", "2024_06_13"]

        #date = ["132", "2024_04_25-26", "2024_04_28-30", "2024_05_06-07", "2024_05_08-09", "2024_05_17-20"]
        date = ["132", "2024_04_25-26", "2024_05_06-07", "2024_05_17-20", "2024_06_11-12", "2024_06_13"]
        # time = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        time = [2, 3, 4, 5]
        as_test = 1
        #del time[as_test-1]
        if flag == "1":
            data_path = [
                data_root + date[time[0]] + "/train_data.pkt",
                data_root + date[time[1]] + "/train_data.pkt",
                data_root + date[time[2]] + "/train_data.pkt",
                data_root + date[time[3]] + "/train_data.pkt",
                #data_root + date[time[4]] + "/train_data.pkt"
            ]
            for i in range(len(data_path)):
                with open(data_path[i], "rb") as f:
                    train_pkt_data.append(pickle.load(f))
        elif flag == "2":
            data_path = [
                data_root + date[time[0]] + "/test_data.pkt",
                data_root + date[time[1]] + "/test_data.pkt",
                data_root + date[time[2]] + "/test_data.pkt",
                data_root + date[time[3]] + "/test_data.pkt",
                #data_root + date[time[4]] + "/test_data.pkt"
            ]
            for i in range(len(data_path)):
                with open(data_path[i], "rb") as f:
                    val_pkt_data.append(pickle.load(f))
        else:
            data_path = [
                data_root + date[as_test] + "/train_data.pkt",
                data_root + date[as_test] + "/test_data.pkt"
            ]
            for i in range(len(data_path)):
                with open(data_path[i], "rb") as f:
                    test_pkt_data.append(pickle.load(f))

        #pkt_data = np.array(pkt_data)
        #idx = [4, 8, 12, 16, 20, 24, 40, 44, 48, 52, 56, 60]
        #data_np = np.zeros((len(pkt_data), 320), dtype=complex)
        #cfo = np.zeros(len(pkt_data), dtype=float)
        label = np.zeros(len(pkt_data), dtype=int)
        for data in pkt_data:
            data_i = data.data[100:100 + 320]
            #a = float(data.cfo)
            # cfo[i] = a
            #data_i = freqCompensation(data_i, a)
            # data_i = phaseCompensation(data_i)
            # my_timer_figure(data_i)
            # cfo_j = freqCfoEstimate(data_i)
            # my_timer_figure(data_i)
            data_np[i] = data_i
            label[i] = data.dev_label

        # IQ
        # data_np = data_norm_np(data_np)
        # stft_data_tensor = torch.from_numpy(data_np)
        # self.stft_data = torch.cat((stft_data_tensor.real.unsqueeze(1).float(), stft_data_tensor.imag.unsqueeze(1).float()), dim=1)

        # FFT
        # data_np = data_norm_np(data_np)
        # data_np = np.fft.fft(data_np)
        # stft_data_tensor = torch.from_numpy(data_np)
        # self.stft_data = torch.cat((stft_data_tensor.real.unsqueeze(1).float(), stft_data_tensor.imag.unsqueeze(1).float()), dim=1)

        # PSD
        # data_np = data_norm_np(data_np)
        # # 对每行数据进行FFT计算
        # fft_result = np.fft.fft(data_np)
        # psd_data = np.zeros_like(data_np)
        # # 计算每行数据的功率谱密度
        # for i in range(data_np.shape[0]):
        #     psd_data[i] = np.abs(fft_result[i]) ** 2 / np.linalg.norm(fft_result[i])
        # stft_data_tensor = torch.from_numpy(psd_data)
        # self.stft_data = stft_data_tensor.unsqueeze(1).float()      # (10000, 1, 256)

        # data_np = data_norm_np(data_np)
        # freq, data_np = signal.welch(data_np, fs=20e6, return_onesided=False)
        # stft_data_tensor = torch.from_numpy(data_np)
        # self.stft_data = stft_data_tensor.unsqueeze(1).float()      # (10000, 1, 256)

        # STFT
        # data_np = data_norm_np(data_np)
        # frequencies, times, data_stft = stft(data_np, nperseg=128, noverlap=115, return_onesided=False)
        # stft_data_tensor = torch.from_numpy(data_stft)
        # self.stft_data = torch.cat((stft_data_tensor.real.unsqueeze(1).float(), stft_data_tensor.imag.unsqueeze(1).float()), dim=1)

        # ChiHis
        # if flag == "1":
        #     data_np = awgn(data_np, range(20, 80))
        # ChannelIndSpectrogramObj = ChannelIndSpectrogram()
        # data_np = ChannelIndSpectrogramObj.channel_ind_spectrogram(data_np)
        # stft_data_tensor = torch.from_numpy(data_np)
        # self.stft_data = stft_data_tensor.unsqueeze(1).float()

        # Dolos
        # data_np = data_norm_np(data_np)
        # sts1 = data_np[:, 16:16 * 5]
        # sts2 = data_np[:, 16 * 5:16 * 9]
        # lts1 = data_np[:, 192:192 + 64]
        # lts2 = data_np[:, 192 + 64:192 + 64 * 2]
        # sts = (sts1+sts2)/2
        # lts = (lts1+lts2)/2
        # rff = np.log(np.abs(np.fft.fft(sts))) - np.log(np.abs(np.fft.fft(lts)))
        # rff = rff[:, idx]
        # stft_data_tensor = torch.from_numpy(rff)
        # self.stft_data = stft_data_tensor.unsqueeze(1).float()

        # CWT
        # data_xiaobo = np.empty((data_np.shape[0], 32, 320), dtype=complex)
        # wavelet = 'gaus1'  # 选择小波函数，这里使用 Morlet 小波
        # for i in range(data_np.shape[0]):
        # # 进行小波变换
        #     coeffs, freqs = pywt.cwt(data_np[i], scales=np.arange(1, 32+1), wavelet=wavelet)
        #     data_xiaobo[i] = coeffs
        # stft_data_tensor = torch.from_numpy(data_xiaobo)
        # self.stft_data = torch.cat((stft_data_tensor.real.unsqueeze(1).float(), stft_data_tensor.imag.unsqueeze(1).float()), dim=1)

        # HHT
        # date = ["132", "2024_04_25-26", "2024_05_06-07", "2024_05_17-20", "2024_06_11-12", "2024_06_13"]
        # time = [1, 2, 3, 4, 5]
        # as_test = 4
        # del time[as_test-1]
        # data_root1 = "/home/data1/WZM/Dataset/temperature/weight_pth/HHT/"
        # save_path = [
        #     data_root1 + date[time[0]] + "_train_data.npy",
        #     data_root1 + date[time[1]] + "_train_data.npy",
        #     data_root1 + date[time[2]] + "_train_data.npy",
        #     data_root1 + date[time[3]] + "_train_data.npy",
        #     data_root1 + date[as_test] + "_train_data.npy",
        #     data_root1 + date[time[0]] + "_test_data.npy",
        #     data_root1 + date[time[1]] + "_test_data.npy",
        #     data_root1 + date[time[2]] + "_test_data.npy",
        #     data_root1 + date[time[3]] + "_test_data.npy",
        #     data_root1 + date[as_test] + "_test_data.npy"
        # ]
        # data_train1 = np.load(save_path[0])
        # data_train2 = np.load(save_path[1])
        # data_train3 = np.load(save_path[2])
        # data_train4 = np.load(save_path[3])
        # data_train5 = np.load(save_path[4])
        # data_test1 = np.load(save_path[5])
        # data_test2 = np.load(save_path[6])
        # data_test3 = np.load(save_path[7])
        # data_test4 = np.load(save_path[8])
        # data_test5 = np.load(save_path[9])
        # if flag == "1":
        #     data_np = np.concatenate((data_train1, data_train2, data_train3, data_train4), axis=0)
        # elif flag == "2":
        #     data_np = np.concatenate((data_test1, data_test2, data_test3, data_test4), axis=0)
        # else:
        #     data_np = np.concatenate((data_train5, data_test5), axis=0)
        # self.stft_data = torch.from_numpy(data_np)

        # emd = EMD()
        # t = np.arange(320)
        # # imfs = emd(data_np)
        # data_np_real = np.real(data_np)
        # data_np_imag = np.imag(data_np)
        # hht_real = np.empty((data_np.shape[0], 5, 320))
        # hht_imag = np.empty((data_np.shape[0], 5, 320))
        # for i in range(data_np.shape[0]):
        #     imfs_real = emd(data_np_real[i], t)
        #     imfs_imag = emd(data_np_imag[i], t)
        #     if len(imfs_real) == 4 or len(imfs_imag) == 4:
        #         zeros = np.full((1, 320), 1e-16)
        #         imfs_real = np.concatenate((imfs_real, zeros), axis=0)
        #         imfs_imag = np.concatenate((imfs_imag, zeros), axis=0)
        #     hht_real[i] = hilbert(imfs_real)[:5]
        #     hht_imag[i] = hilbert(imfs_imag)[:5]
        # data_norm_real = torch.from_numpy(hht_real).float()
        # data_norm_imag = torch.from_numpy(hht_imag).float()
        # print("123")
        # self.stft_data = torch.cat((data_norm_real.unsqueeze(1), data_norm_imag.unsqueeze(1)), dim=1)  # (10000, 2, 5, 320)

        # EMD
        # date = ["132", "2024_04_25-26", "2024_05_06-07", "2024_05_17-20", "2024_06_11-12", "2024_06_13"]
        # time = [1, 2, 3, 4, 5]
        # as_test = 4
        # del time[as_test-1]
        # data_root1 = "/home/data1/WZM/Dataset/temperature/weight_pth/EMD/"
        # save_path = [
        #     data_root1 + date[time[0]] + "_train_data.npy",
        #     data_root1 + date[time[1]] + "_train_data.npy",
        #     data_root1 + date[time[2]] + "_train_data.npy",
        #     data_root1 + date[time[3]] + "_train_data.npy",
        #     data_root1 + date[as_test] + "_train_data.npy",
        #     data_root1 + date[time[0]] + "_test_data.npy",
        #     data_root1 + date[time[1]] + "_test_data.npy",
        #     data_root1 + date[time[2]] + "_test_data.npy",
        #     data_root1 + date[time[3]] + "_test_data.npy",
        #     data_root1 + date[as_test] + "_test_data.npy"
        # ]
        # data_train1 = np.load(save_path[0])
        # data_train2 = np.load(save_path[1])
        # data_train3 = np.load(save_path[2])
        # data_train4 = np.load(save_path[3])
        # data_train5 = np.load(save_path[4])
        # data_test1 = np.load(save_path[5])
        # data_test2 = np.load(save_path[6])
        # data_test3 = np.load(save_path[7])
        # data_test4 = np.load(save_path[8])
        # data_test5 = np.load(save_path[9])
        # if flag == "1":
        #     data_np = np.concatenate((data_train1, data_train2, data_train3, data_train4), axis=0)
        # elif flag == "2":
        #     data_np = np.concatenate((data_test1, data_test2, data_test3, data_test4), axis=0)
        # else:
        #     data_np = np.concatenate((data_train5, data_test5), axis=0)
        # self.stft_data = torch.from_numpy(data_np)

        # emd = EMD()
        # t = np.arange(320)
        # # imfs = emd(data_np)
        # data_np_real = np.real(data_np)
        # data_np_imag = np.imag(data_np)
        # emd_real = np.empty((data_np.shape[0], 5, 320))
        # emd_imag = np.empty((data_np.shape[0], 5, 320))
        # for i in range(data_np.shape[0]):
        #     imfs_real = emd(data_np_real[i], t)
        #     imfs_imag = emd(data_np_imag[i], t)
        #     # a = len(imfs_real)
        #     if len(imfs_real)==4 or len(imfs_imag)==4:
        #         zeros = np.full((1, 320), 1e-16)
        #         imfs_real = np.concatenate((imfs_real, zeros), axis=0)
        #         imfs_imag = np.concatenate((imfs_imag, zeros), axis=0)
        #     emd_real[i] = imfs_real[:5]
        #     emd_imag[i] = imfs_imag[:5]
        # data_norm_real = torch.from_numpy(emd_real).float()
        # data_norm_imag = torch.from_numpy(emd_imag).float()
        # print("123")
        # self.stft_data = torch.cat((data_norm_real.unsqueeze(1), data_norm_imag.unsqueeze(1)), dim=1)    # (10000, 2, 5, 320)

        # Gabor
        # data_np = data_np[:, :, np.newaxis]
        # # 定义 Gabor 参数
        # frequency = 0.2
        # theta = 0
        # sigma_x = None
        # sigma_y = None
        # n_stds = 3
        # # 创建一个空数组用于保存结果
        # gabor_result_real = np.zeros_like(data_np)
        # gabor_result_imag = np.zeros_like(data_np)
        # # 遍历每行数据并应用 Gabor 变换
        # for i in range(data_np.shape[0]):
        #     row_data = data_np[i]  # 获取每条数据
        #     filtered_row_real, _ = filters.gabor(np.real(row_data), frequency=frequency, theta=theta, sigma_x=sigma_x, sigma_y=sigma_y, n_stds=n_stds)  # 应用 Gabor 滤波器
        #     filtered_row_imag, _ = filters.gabor(np.real(row_data), frequency=frequency, theta=theta, sigma_x=sigma_x, sigma_y=sigma_y, n_stds=n_stds)  # 应用 Gabor 滤波器
        #     gabor_result_real[i] = filtered_row_real  # 保存结果到 gabor_result 中
        #     gabor_result_imag[i] = filtered_row_imag
        #     # my_timer_figure(gabor_result[i])
        # data_norm_real = torch.from_numpy(gabor_result_real).squeeze(2).float()
        # data_norm_imag = torch.from_numpy(gabor_result_imag).squeeze(2).float()
        # self.stft_data = torch.cat((data_norm_real.unsqueeze(1), data_norm_imag.unsqueeze(1)), dim=1)  # (10000, 2, 320)

        # VMD
        # data = np.zeros((data_np.shape[0], 5, 320))
        # alpha = 2000
        # tau = 0  # tau 噪声容限，即允许重构后的信号与原始信号有差别。
        # K = 5  # K 分解模态（IMF）个数
        # DC = 0  # DC 若为0则让第一个IMF为直流分量/趋势向量
        # init = 1  # init 指每个IMF的中心频率进行初始化。当初始化为1时，进行均匀初始化。
        # tol = 1e-7  # 控制误差大小常量，决定精度与迭代次数
        # for i in range(data_np.shape[0]):
        #     # 对信号进行VMD分解
        #     data[i], u_hat, omega = VMD(data_np[i], alpha, tau, K, DC, init, tol)
        # data_norm_real = torch.from_numpy(np.real(data)).float()
        # data_norm_imag = torch.from_numpy(np.imag(data)).float()
        # self.stft_data = torch.cat((data_norm_real.unsqueeze(1), data_norm_imag.unsqueeze(1)), dim=1)

        # EPS
        # data_np_real = np.zeros(data_np.shape)
        # data_np_imag = np.zeros(data_np.shape)
        # hilbert_filter = firwin(101, 0.5)
        # for i in range(data_np.shape[0]):
        #     envelope_I = np.abs(np.convolve(np.real(data_np[i]), hilbert_filter, mode='same'))
        #     envelope_Q = np.abs(np.convolve(np.imag(data_np[i]), hilbert_filter, mode='same'))
        #     envelope_I = np.real(data_np[i]) + envelope_I * 1j
        #     envelope_Q = np.imag(data_np[i]) + envelope_Q * 1j
        #     envelope_I_dc_removed = envelope_I - np.mean(envelope_I, axis=0)
        #     envelope_Q_dc_removed = envelope_Q - np.mean(envelope_Q, axis=0)
        #     # 计算功率谱密度（PSD）
        #     psd_I = np.abs(np.fft.fft(envelope_I_dc_removed, axis=0)) ** 2 / np.linalg.norm(np.fft.fft(envelope_I_dc_removed, axis=0))
        #     psd_Q = np.abs(np.fft.fft(envelope_Q_dc_removed, axis=0)) ** 2 / np.linalg.norm(np.fft.fft(envelope_Q_dc_removed, axis=0))
        #     # 归一化功率谱
        #     freq = np.fft.fftfreq(envelope_I_dc_removed.shape[0], d=1)  # 频率轴
        #     data_np_real[i] = psd_I / np.max(psd_I, axis=0)
        #     data_np_imag[i] = psd_Q / np.max(psd_Q, axis=0)
        # data_norm_real = torch.from_numpy(data_np_real).float()
        # data_norm_imag = torch.from_numpy(data_np_imag).float()
        # self.stft_data = torch.cat((data_norm_real.unsqueeze(1), data_norm_imag.unsqueeze(1)), dim=1)   # (10000, 2, 320)

        # CLPS
        # data_np = data_norm_np(data_np)
        # p_log = np.log(np.abs(np.fft.fft(data_np)))
        # clps = 2 * 20e6 * (p_log - np.mean(p_log))
        # clps = torch.from_numpy(clps)
        # self.stft_data = clps.unsqueeze(1).float()

        # Ours
        data_np = data_norm_np(data_np)
        stft_data_tensor = torch.from_numpy(data_np)
        self.stft_data = stft_data_tensor

        self.label = torch.LongTensor(label)
        self.label_np = label

    def __getitem__(self, index):
        # raw_data = self.raw_data[index]
        # fft_data = self.fft_data[index]
        stft_data = self.stft_data[index]
        label = self.label[index]
        # cfo = self.cfo[index]
        return stft_data, label

    def __len__(self):
        return len(self.label)


class myDataset6(Dataset):
    def __init__(self, data_path=None, flag=None, flag2=None):
        pkt_data = []
        data_root = "/home/data1/WZM/Dataset/temperature//"
        # date = ["132", "2024_04_25-26", "2024_04_28-30", "2024_05_06-07", "2024_05_08-09", "2024_05_17-20", "2024_06_03-04", "2024_06_05-06", "2024_06_11-12", "2024_06_13"]
        # date = ["132", "2024_04_25-26", "2024_04_28-30", "2024_05_06-07", "2024_05_08-09", "2024_05_17-20"]
        # date = ["132", "2024_04_25-26", "2024_05_06-07", "2024_05_17-20", "2024_06_11-12", "2024_06_13"]
        test_path = ["2024_06_03-04", "2024_06_05-06", "2024_06_11-12", "2024_06_13"]
        # time = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        # time = [1, 2, 3, 4, 5]
        # as_test = 4
        # del time[as_test-1]
        if flag == "0":
            data_path = [
                "/home/data1/WZM/Dataset/temperature/2024_04_25-26/train_data.pkt",
                "/home/data1/WZM/Dataset/temperature/2024_04_28-30/train_data.pkt",
                "/home/data1/WZM/Dataset/temperature/2024_05_06-07/train_data.pkt",
                "/home/data1/WZM/Dataset/temperature/2024_05_08-09/train_data.pkt",
                "/home/data1/WZM/Dataset/temperature/2024_05_17-20/train_data.pkt"
                # data_root + date[time[4]] + "/train_data.pkt"
            ]
            for i in range(len(data_path)):
                with open(data_path[i], "rb") as f:
                    pkt_data.extend(pickle.load(f))
        elif flag == "1":
            data_path = [
                "/home/data1/WZM/Dataset/temperature/" + test_path[0] + "/train_data.pkt",
                "/home/data1/WZM/Dataset/temperature/" + test_path[0] + "/test_data.pkt"
            ]
            for i in range(len(data_path)):
                with open(data_path[i], "rb") as f:
                    pkt_data.extend(pickle.load(f))
        elif flag == "2":
            data_path = [
                "/home/data1/WZM/Dataset/temperature/" + test_path[1] + "/train_data.pkt",
                "/home/data1/WZM/Dataset/temperature/" + test_path[1] + "/test_data.pkt"
            ]
            for i in range(len(data_path)):
                with open(data_path[i], "rb") as f:
                    pkt_data.extend(pickle.load(f))
        elif flag == "3":
            data_path = [
                "/home/data1/WZM/Dataset/temperature/" + test_path[2] + "/train_data.pkt",
                "/home/data1/WZM/Dataset/temperature/" + test_path[2] + "/test_data.pkt"
            ]
            for i in range(len(data_path)):
                with open(data_path[i], "rb") as f:
                    pkt_data.extend(pickle.load(f))
        elif flag == "4":
            data_path = [
                "/home/data1/WZM/Dataset/temperature/" + test_path[3] + "/train_data.pkt",
                "/home/data1/WZM/Dataset/temperature/" + test_path[3] + "/test_data.pkt"
            ]
            for i in range(len(data_path)):
                with open(data_path[i], "rb") as f:
                    pkt_data.extend(pickle.load(f))
        elif flag == "5":
            data_path = [
                "/home/data1/WZM/Dataset/temperature/2024_04_25-26/test_data.pkt",
                "/home/data1/WZM/Dataset/temperature/2024_04_28-30/test_data.pkt",
                "/home/data1/WZM/Dataset/temperature/2024_05_06-07/test_data.pkt",
                "/home/data1/WZM/Dataset/temperature/2024_05_08-09/test_data.pkt",
                "/home/data1/WZM/Dataset/temperature/2024_05_17-20/test_data.pkt"
            ]
            for i in range(len(data_path)):
                with open(data_path[i], "rb") as f:
                    pkt_data.extend(pickle.load(f))
        pkt_data = np.array(pkt_data)
        idx = [4, 8, 12, 16, 20, 24, 40, 44, 48, 52, 56, 60]
        data_np = np.zeros((len(pkt_data), 320), dtype=complex)
        cfo = np.zeros(len(pkt_data), dtype=float)
        label = np.zeros(len(pkt_data), dtype=int)
        for i, data in enumerate(pkt_data):
            data_i = data.data[100:100 + 320]
            a = float(data.cfo)
            # cfo[i] = a
            # data_i = freqCompensation(data_i, a)
            # data_i = phaseCompensation(data_i)
            # my_timer_figure(data_i)
            # cfo_j = freqCfoEstimate(data_i)
            # my_timer_figure(data_i)
            data_np[i] = data_i
            label[i] = data.dev_label
        data_new = []
        label_new = []

        if flag2 == 'test':
            for n in range(10):
                idx_n = np.where(label == n)
                data_n = data_np[idx_n]
                label_n = label[idx_n]
                random_idx = np.random.choice(data_n.shape[0], 1000, replace=False)
                data_new.extend(data_n[random_idx])
                label_new.extend(label_n[random_idx])
            data_np = np.array(data_new)
            label = np.array(label_new)

        # IQ
        # stft_data_tensor = torch.from_numpy(data_np)
        # self.stft_data = torch.cat((stft_data_tensor.real.unsqueeze(1).float(), stft_data_tensor.imag.unsqueeze(1).float()), dim=1)

        # Ours
        data_np = data_norm_np(data_np)
        stft_data_tensor = torch.from_numpy(data_np)
        self.stft_data = stft_data_tensor

        self.label = torch.LongTensor(label)
        self.label_np = label

    def __getitem__(self, index):
        # raw_data = self.raw_data[index]
        # fft_data = self.fft_data[index]
        stft_data = self.stft_data[index]
        label = self.label[index]
        # cfo = self.cfo[index]
        return stft_data, label

    def __len__(self):
        return len(self.label)


def phaseCompensation(preamble):
    # 相偏补偿
    short_16, short, long, local_preamble = localSequenceGenerate()
    angle_i = np.angle(preamble * np.conj(local_preamble))
    # my_timer_figure(np.abs(np.fft.fft(preamble)))
    preamble = preamble * np.exp(1j * (-angle_i))
    # angle_j = np.angle(preamble * np.conj(local_preamble))
    # my_timer_figure(np.abs(np.fft.fft(preamble)))
    return preamble


def freqCfoEstimate_(preamble, fs=20e6):
    # 粗频偏补偿
    delta_i = []
    for n in range(7):  # 第n个周期的第L个采样点
        sita = []
        for L in range(16):
            idx = 16+16*n+L
            sita.append(np.angle(preamble[idx+16] * np.conj(preamble[idx])))
        delta_i.append(np.mean(sita))
    delta = np.mean(delta_i)
    delta_f = delta * fs / (2 * np.pi * 16)

    # 细频偏补偿
    sita_long = []
    for L in range(64):
        idx_long = 160+32+L
        sita_long.append(np.angle(preamble[idx_long] * np.conj(preamble[idx_long+64])))
    delta_long = np.mean(sita_long)
    delta_f_long = delta_long * fs / (2 * np.pi * 64)
    delta_freq = delta_f+delta_f_long
    return delta_freq


def localSequenceGenerate():
    # STS频域表示，频点为 -32~31，此处将52个频点外的零补全。
    S = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1 + 1j, 0, 0, 0, -1 - 1j, 0, 0, 0, 1 + 1j, 0, 0, 0, -1 - 1j, 0, 0, 0, -1 - 1j, 0, 0, 0, 1 + 1j, 0, 0, 0, 0, 0, 0, 0, -1 - 1j, 0, 0, 0, -1 - 1j, 0, 0, 0, 1 + 1j, 0, 0, 0, 1 + 1j, 0, 0, 0, 1 + 1j, 0, 0, 0, 1 + 1j, 0, 0, 0, 0, 0, 0, 0])
    # LTS频域表示，频点为 - 32~31，此处将52个频点外的零补全。
    L = np.array([0, 0, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 0, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    # 保证OFDM符号的功率值稳定
    S = np.sqrt(13 / 6) * S
    S_shifted = np.fft.fftshift(S)
    # my_timer_figure1(np.abs(S_shifted))
    # 通过IFFT函数将STS频域的频点顺序调整为正频率（0~31）、负频率（-32~-1）
    short_16 = np.fft.ifft(S_shifted)[:16]
    # my_timer_figure(short_16)
    short = np.tile(short_16, 10)
    # short[0] = 0.5*short[0]
    # short[-1] = 0.5 * short[-1]
    # my_timer_figure(short)
    L_shifted = np.fft.fftshift(L)
    # my_timer_figure1(np.abs(L))
    long_cp = np.fft.ifft(L_shifted)
    long1 = long_cp[32:]
    long2 = long_cp
    long = np.concatenate((long1, long2, long2))
    preamble = np.concatenate((short, long))
    # 第161个数据加窗处理
    preamble[160] = preamble[160] * 0.5 + preamble[0] * 0.5

    # 第一个数据加窗处理
    preamble[0] = preamble[0] * 0.5

    return short_16, short, long, preamble


# Min-Max 归一化函数
def min_max_normalize(matrix):
    min_vals = np.min(matrix, axis=1, keepdims=True)
    max_vals = np.max(matrix, axis=1, keepdims=True)
    normalized_matrix = (matrix - min_vals) / (max_vals - min_vals)
    return normalized_matrix


def data_norm_np(data_np):
    data_norm = np.zeros(data_np.shape, dtype=complex)
    for i in range(data_np.shape[0]):
        sig_amplitude = np.abs(data_np[i])
        rms = np.sqrt(np.mean(sig_amplitude ** 2))
        data_norm[i] = data_np[i] / rms
    return data_norm

def data_norm(data_tensor):
    data_norm = torch.zeros_like(data_tensor, dtype=torch.cfloat)
    for i in range(data_tensor.shape[0]):
        sig_amplitude = torch.abs(data_tensor[i])
        rms = torch.sqrt(torch.mean(sig_amplitude ** 2))
        data_norm[i] = data_tensor[i] / rms
    return data_norm


def normalized_2d(data):
    norm_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        norm_data[i] = (data[i]-data[i].min()) / (data[i].max()-data[i].min())
    return norm_data


# 时域波形图
def my_timer_figure(data):
    plt.figure(figsize=(16, 8))
    plt.plot(data)
    plt.title('EPS')
    plt.xlabel('time')
    plt.ylabel('AMP')
    plt.show()


class ChannelIndSpectrogram():
    def __init__(self, ):
        pass

    def _normalization(self, data):
        ''' Normalize the signal.'''
        s_norm = np.zeros(data.shape, dtype=complex)

        for i in range(data.shape[0]):
            sig_amplitude = np.abs(data[i])
            rms = np.sqrt(np.mean(sig_amplitude ** 2))
            s_norm[i] = data[i] / rms

        return s_norm

    def _spec_crop(self, x):
        '''Crop the generated channel independent spectrogram.'''
        num_row = x.shape[0]
        x_cropped = x[round(num_row * 0.3):round(num_row * 0.7)-1]

        return x_cropped

    def _gen_single_channel_ind_spectrogram(self, sig, win_len=64, overlap=32):
        '''
        _gen_single_channel_ind_spectrogram converts the IQ samples to a channel
        independent spectrogram according to set window and overlap length.

        INPUT:
            SIG is the complex IQ samples.

            WIN_LEN is the window length used in STFT.

            OVERLAP is the overlap length used in STFT.

        RETURN:

            CHAN_IND_SPEC_AMP is the genereated channel independent spectrogram.
        '''
        # # 进行小波变换
        # wavelet = 'morl'  # 选择小波函数，这里使用 Morlet 小波
        # coeffs, freqs = pywt.cwt(sig, scales=np.arange(1, 128), wavelet=wavelet)
        #
        # # 绘制时频图
        # plt.figure(figsize=(10, 6))
        # plt.imshow(np.abs(coeffs), aspect='auto', extent=[0, len(sig), freqs[-1], freqs[0]])
        # plt.title("小波变换时频图")
        # plt.xlabel("时间")
        # plt.ylabel("频率")
        # plt.colorbar(label="振幅")
        # plt.grid(True)
        # plt.show()
        # # 计算数据的频谱
        # spectrum = np.fft.fft(sig)
        #
        # # 计算频率轴
        # n = len(sig)
        # frequency = np.fft.fftfreq(n)
        #
        # # 绘制频谱图
        # plt.figure(figsize=(10, 6))
        # plt.plot(frequency, np.abs(spectrum))
        # plt.title("频谱图")
        # plt.xlabel("频率 (Hz)")
        # plt.ylabel("振幅")
        # plt.grid(True)
        # plt.show()
        # Short-time Fourier transform (STFT).
        f, t, spec = signal.stft(sig,
                                 window='boxcar',
                                 nperseg=win_len,
                                 noverlap=overlap,
                                 nfft=win_len,
                                 return_onesided=False,
                                 padded=False,
                                 boundary=None)

        # FFT shift to adjust the central frequency.
        spec = np.fft.fftshift(spec, axes=0)

        # Generate channel independent spectrogram.
        chan_ind_spec = spec[:, 1:] / spec[:, :-1]

        # Take the logarithm of the magnitude.
        chan_ind_spec_amp = np.log10(np.abs(chan_ind_spec) ** 2)
        # chan_ind_spec_amp = np.log10(chan_ind_spec)

        return chan_ind_spec_amp

    def channel_ind_spectrogram(self, data):
        '''
        channel_ind_spectrogram converts IQ samples to channel independent
        spectrograms.

        INPUT:
            DATA is the IQ samples.

        RETURN:
            DATA_CHANNEL_IND_SPEC is channel independent spectrograms.
        '''

        # Normalize the IQ samples.
        data = self._normalization(data)

        # Calculate the size of channel independent spectrograms.
        num_sample = data.shape[0]      #样本数量,17个
        # num_row = int(256 * 0.4)
        num_row = int(64 * 0.4)

        # num_column = int(np.floor((data.shape[1] - 256) / 128 + 1) - 1)
        num_column = int(np.floor((data.shape[1] - 64) / 32 + 1) - 1)
        data_channel_ind_spec = np.zeros([num_sample, num_row, num_column])

        # Convert each packet (IQ samples) to a channel independent spectrogram.
        for i in range(num_sample):
            chan_ind_spec_amp = self._gen_single_channel_ind_spectrogram(data[i])
            # # 创建一个新的图像
            # plt.figure()
            # # 使用 imshow 函数来显示 STFT 的结果
            # plt.imshow(np.abs(chan_ind_spec_amp), aspect='auto', origin='lower')
            # # 添加颜色条
            # plt.colorbar()
            # # 设置标题和坐标轴标签
            # plt.title('STFT Magnitude')
            # plt.xlabel('Time')
            # plt.ylabel('Frequency')
            # # 显示图像
            # plt.show()
            # print(np.array(chan_ind_spec_amp).shape)
            chan_ind_spec_amp = self._spec_crop(chan_ind_spec_amp)
            # print(np.array(chan_ind_spec_amp).shape)
            data_channel_ind_spec[i, :, :] = chan_ind_spec_amp

        return data_channel_ind_spec

def awgn(data, snr_range):
    pkt_num = data.shape[0]
    SNRdB = uniform(snr_range[0], snr_range[-1], pkt_num)
    for pktIdx in range(pkt_num):
        s = data[pktIdx]
        # SNRdB = uniform(snr_range[0],snr_range[-1])
        SNR_linear = 10 ** (SNRdB[pktIdx] / 10)
        P = sum(abs(s) ** 2) / len(s)
        N0 = P / SNR_linear
        n = np.sqrt(N0 / 2) * (standard_normal(len(s)) + 1j * standard_normal(len(s)))
        data[pktIdx] = s + n
    return data