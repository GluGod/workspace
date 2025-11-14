import numpy as np
import torch
import torch.utils.data as Data
from torch.utils.data import Dataset
import pickle

from TTA.DPL.dataloader import Mydataset
from getMyDataset import *




def data_norm_np(data_np):
    data_norm = np.zeros(data_np.shape, dtype=complex)
    for i in range(data_np.shape[0]):
        sig_amplitude = np.abs(data_np[i])
        rms = np.sqrt(np.mean(sig_amplitude ** 2))
        data_norm[i] = data_np[i] / rms
    return data_norm



class Wifi(object):
    def __init__(self, args, num_workers=48, batch_size=256):
        # date = ["2024_04_25-26", "2024_05_06-07", "2024_05_17-20", "2024_06_03-04","2024_06_11-12", "2024_06_13"]
        # date = ["2024_04_25-26", "2024_04_28-30","2024_05_06-07", "2024_05_08-09","2024_05_17-20"]

        date = ["2024_04_25-26", "2024_05_06-07", "2024_06_03-04","2024_05_17-20", "2024_04_28-30"]

        split = None#'multiday'

        if split == 'multiday':
            path2 = "/data1/hw/WK/wifi信号数据集/" + date[0] + "/test_data.pkt"
            test_data, test_label = self.data_read(path2)
            testset_dataset = Data.TensorDataset(torch.from_numpy(test_data), torch.LongTensor(test_label))
            # testset_dataset = myDataset4(path2)

            self.test_loader = torch.utils.data.DataLoader(
                testset_dataset, batch_size=batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=True,
            )
            data = []
            label = []
            for i in [1,2,3,4]:
                path = "/data1/hw/WK/wifi信号数据集/" + date[i] + "/train_data.pkt"
                d, l = self.data_read(path)
                data.append(d)
                label.append(l)

            data = np.concatenate(data, axis=0)  # 沿第一个维度（样本维度）合并
            label = np.concatenate(label, axis=0)  # 同上
            train_dataset = Data.TensorDataset(torch.from_numpy(data), torch.LongTensor(label))
            # train_dataset = myDataset4(path1)

            self.train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=True,
            )

        else:
            path1 = "/data1/hw/WK/wifi信号数据集/" + date[0] + "/train_data.pkt"
            data, label = self.data_read(path1)
            print(data.shape)
            train_dataset = Data.TensorDataset(torch.from_numpy(data), torch.LongTensor(label))
            # train_dataset = myDataset4(path1)

            self.train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=True,
            )

            path2 = "/data1/hw/WK/wifi信号数据集/" + date[1] + "/test_data.pkt"
            test_data, test_label = self.data_read(path2)
            testset_dataset = Data.TensorDataset(torch.from_numpy(test_data), torch.LongTensor(test_label))
            # testset_dataset = myDataset4(path2)

            self.test_loader = torch.utils.data.DataLoader(
                testset_dataset, batch_size=batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=True,
            )

    def read_pkt_file(self, file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data

    def data_read(self, path):
        # 读取.pkt文件
        data = self.read_pkt_file(path)
        data_np = np.zeros((len(data), 320), dtype=complex)
        train_label = np.zeros(len(data), dtype=int)
        for i, data in enumerate(data):
            data_i = data.data[100:100 + 320]
            data_np[i] = data_i
            train_label[i] = data.dev_label

        # data = np.load(root + path)
        train_data = data_np
        train_data = data_norm_np(train_data)
        #local_data, shot, long = localSequenceGenerate()
        # train_data = data_norm(train_data - data_norm(np.concatenate((shot, long), axis=0)))

        # train_data_real, train_data_imag = np.real(train_data).reshape(len(train_data), 1, 320), np.imag(
        #     train_data).reshape(len(train_data), 1, 320)
        # train_data = np.concatenate((train_data_real, train_data_imag), axis=1)
        train_label = train_label
        return train_data, train_label