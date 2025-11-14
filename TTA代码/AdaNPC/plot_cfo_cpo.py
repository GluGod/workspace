import numpy as np

import numpy as np
import random
from collections import defaultdict
import pickle

def read_pkt_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def data_read(rang):
    # 初始化存储列表
    train_data = []
    train_label = []

    for j in rang:

        date = ["2024_04_25-26", "2024_05_06-07", "2024_05_17-20", "2024_06_03-04", "2024_06_11-12", "2024_06_13"]
        path1 = "/data1/hw/WK/wifi信号数据集/" + date[j] + "/train_data.pkt"

        # 读取.pkt文件
        data = read_pkt_file(path1)

        # 使用字典按(设备, App)分组存储样本
        grouped_data = defaultdict(list)

        # 第一次遍历：按(设备, App)分组
        for sample in data:
            device = sample.dev_label
            app = sample.app_label
            grouped_data[(device[0], app)].append(sample)

        # 提取每种组合的30个样本
        for (device, app), samples in grouped_data.items():
            # 如果该组的样本数不足30，打印警告
            if len(samples) < 30:
                print(f"警告: 设备 {device} 的 App {app} 只有 {len(samples)} 个样本，少于要求的30个")
            device_range = [0,1]
            app_range = [0,1]
            if device in device_range and app in app_range:
                # 随机选择30个样本（如果可用）
                # selected = random.sample(samples, min(15, len(samples)))
                selected = samples[100:110]
                # 存储选中的样本
                for sample in selected:
                    train_data.append(sample)
                    train_label.append('Date'+str(j)+'_Dev'+str(device)+'_T'+str(app))

    return train_data, train_label


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

def data_norm_np(data_np):
    data_norm = np.zeros(data_np.shape, dtype=complex)
    for i in range(data_np.shape[0]):
        sig_amplitude = np.abs(data_np[i])
        rms = np.sqrt(np.mean(sig_amplitude ** 2))
        data_norm[i] = data_np[i] / rms
    return data_norm


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd

# 生成颜色映射
def generate_colormap(labels):
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    return dict(zip(unique_labels, colors))


def plot_cfo_cpo(samples, labels):
    # 初始化存储列表
    cfos = []
    cpos = []
    label_colors = []

    # 生成颜色映射和标记映射
    colormap = generate_colormap(labels)
    # 为6个标签定义不同的标记
    markers = ['o', '^', 's', 'D', '*', 'P']  # 圆形、三角形、方形、菱形、五角星、加号五角星

    # 创建标签到标记的映射
    unique_labels = list(set(labels))
    markermap = {label: markers[i % len(markers)] for i, label in enumerate(unique_labels)}

    # 遍历所有样本
    for i, sample in enumerate(samples):
        # 提取数据段
        data_i = sample.data[100:100 + 320]

        # 获取CFO (载波频率偏移)
        cfo = float(sample.cfo)
        cfos.append(cfo)

        # 获取标签
        label = labels[i]
        label_colors.append(colormap[label])

        # 计算CPO (载波相位偏移)
        short_16, short, long, local_preamble = localSequenceGenerate()
        # 计算接收信号与本地前导码之间的相位差
        phase_diff = np.angle(data_i * np.conj(local_preamble))
        # 取平均作为CPO估计
        cpo = np.mean(phase_diff)
        cpos.append(cpo)

    # 转换为NumPy数组
    cfos = np.array(cfos)
    cpos = np.array(cpos)

    # 将数据保存到CSV
    df = pd.DataFrame({
        'CFO_Hz': cfos,
        'CPO_rad': cpos,
        'Label': labels
    })

    # 保存到CSV文件
    csv_filename = 'cfo_cpo_data.csv'
    df.to_csv(csv_filename, index=False)
    print(f"数据已保存到 {csv_filename}")

    # # 计算目标尺寸（厘米转英寸）
    # width_cm = 22.4
    # height_cm = 16.7
    # width_inch = width_cm / 2.54
    # height_inch = height_cm / 2.54

    # # 创建图形（指定精确尺寸）
    # plt.figure(figsize=(width_inch, height_inch), dpi=300)
    #
    # # 设置全局字体为 Times New Roman
    # plt.rcParams['font.family'] = 'Times New Roman'
    #
    # # 设置全局字体加粗
    # plt.rcParams['font.weight'] = 'bold'
    # plt.rcParams['axes.labelweight'] = 'bold'
    # plt.rcParams['axes.titleweight'] = 'bold'
    #
    # # 分别绘制每个标签的数据点
    # for label in unique_labels:
    #     # 获取当前标签对应的索引
    #     indices = [i for i, lbl in enumerate(labels) if lbl == label]
    #     # 绘制当前标签的数据点
    #     plt.scatter(cfos[indices], cpos[indices],
    #                 c=[colormap[label]] * len(indices),
    #                 marker=markermap[label],
    #                 s=600,  # 调整标记大小以适应新尺寸
    #                 alpha=0.8,
    #                 edgecolors='w',  # 白色边框增加对比度
    #                 linewidths=0.7,
    #                 label=f'{label}')
    #
    # # 添加标签 - Times New Roman 字体
    # plt.xlabel('CFO (Hz)', fontsize=32, fontweight='bold')
    # plt.ylabel('CPO (rad)', fontsize=32, fontweight='bold')
    # plt.grid(True, linestyle='--', alpha=0.7)
    #
    # # 添加图例 - Times New Roman 字体
    # leg = plt.legend(loc='upper right', fontsize=24)
    # for text in leg.get_texts():
    #     text.set_fontweight('bold')
    #     text.set_fontname('Times New Roman')
    # # # 创建两排三列的图例
    # # ncol = 3  # 每行3个图例项
    # # leg = plt.legend(ncol=ncol,
    # #                  loc='upper center',
    # #                  bbox_to_anchor=(0.5, 1.15),  # 在图像上方15%的位置
    # #                  fontsize=24,
    # #                  frameon=False)  # 去掉图例边框
    # # # 设置图例文本格式
    # # for text in leg.get_texts():
    # #     text.set_fontweight('bold')
    # #     text.set_fontname('Times New Roman')
    # #     text.set_fontsize(24)
    # #
    # # # 调整上边距为图例留出空间
    # # plt.subplots_adjust(top=0.85)
    #
    # # 设置刻度标签 - Times New Roman 字体
    # plt.xticks(fontsize=26, fontweight='bold', fontname='Times New Roman')
    # plt.yticks(fontsize=26, fontweight='bold', fontname='Times New Roman')
    #
    # # 设置刻度参数 - Times New Roman 字体
    # plt.tick_params(axis='both', which='major', labelsize=26)
    #
    # # 确保所有文本元素使用 Times New Roman
    # for item in ([plt.gca().xaxis.label, plt.gca().yaxis.label] +
    #              plt.gca().get_xticklabels() + plt.gca().get_yticklabels()):
    #     item.set_fontname('Times New Roman')
    #     item.set_fontweight('bold')
    #
    # # 调整边框和布局
    # plt.subplots_adjust(left=0.08, right=0.95, bottom=0.08, top=0.95)
    #
    # # 保存为PDF格式
    # plt.savefig('cfo_cpo_plot.pdf', format='pdf', bbox_inches='tight', dpi=300)
    #
    # # 显示图形
    # plt.tight_layout()
    # plt.show()

    return cfos, cpos, label_colors


# 主程序
rang = [1,2]
samples, labels = data_read(rang)
samples = samples[:6*10]
labels = labels[:6*10]

# 绘制CFO vs CPO图
cfos, cpos, label_colors = plot_cfo_cpo(samples, labels)

