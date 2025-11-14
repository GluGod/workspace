import os

from getMyDataset import plot_IQimage

from model import FeatureNet,CrossNet, WGCN,Neg_pro
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from util import  AverageMeter,count_acc
from torch.utils.data import DataLoader
import numpy as np
import torch.utils.data as Data
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_curve, roc_auc_score
from util import *
from utils import *
from copy import deepcopy
from tqdm import tqdm

class MetaTrainer(object):
    def __init__(self, args):
        os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpus)
        self.model = FeatureNet(args)
        self.model.apply(weights_init)
        self.args = args
        if torch.cuda.is_available():
                self.model = self.model.cuda()

    def replace_base_fc(self, trainloader):
        self.model.eval()
        embedding_list = []
        label_list = []
        # data_list=[]
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                data, label = [_.cuda() for _ in batch]
                data = self.process_rff(data)
                embedding = self.model(data)

                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)

        proto_list = []

        for class_index in range(10):
            data_index = (label_list == class_index).nonzero()
            embedding_this = embedding_list[data_index.squeeze(-1)]
            embedding_this = embedding_this.mean(0)
            proto_list.append(embedding_this)

        proto_list = torch.stack(proto_list, dim=0)

        #self.model.cls.weight.data[:237] = proto_list
        return proto_list

    def freq_augmentation(self,data, freq):
        # cfo_det1 = (torch.rand(data.size(0), 1)-0.5)*2*2000
        cfo_det = (torch.rand(data.size(0), 1) * freq).cuda()
        # noise = (torch.rand(data.size(0), 1)* 0.1).to(device)
        # raw_data1 = torch.complex(data[:, 0, :], data[:, 1, :]).squeeze(-1)
        # raw_data2 = torch.complex(data[:, 2, :], data[:, 3, :]).squeeze(-1)
        t = torch.arange(320).cuda() / 20e6
        data1 = data * torch.exp(1j * 2 * torch.pi * cfo_det * t)
        # data2 = raw_data2 * torch.exp(1j * 2 * torch.pi * cfo_det * t)
        return data1

    def phase_augmentation(self,data, phase):
        phase_det = (torch.rand(data.size(0), 1) * phase).cuda()


        data1 = data * torch.exp(1j * (-phase_det))
        return data1

    def process_rff1(self,data):

        sts1 = data[:, 16:16 * 5]
        sts2 = data[:, 16 * 5:16 * 9]
        lts1 = data[:, 192:192 + 64]
        lts2 = data[:, 192 + 64:192 + 64 * 2]
        # sts = torch.cat((sts1, sts2), dim=-1)
        # lts = torch.cat((lts1, lts2), dim=-1)
        # sts0 = data[:, :16*4], data[:, 160-32:160+32]
        rff3 = torch.log(torch.fft.fft(data[:, 16 * 0:16 * 4])) - torch.log(torch.fft.fft(data[:, 160 - 32:160 + 32]))
        # data = torch.fft.fft(data)
        # my_timer_figure(data[2].cpu().numpy())
        # data = torch.fft.fftshift(data)
        # my_timer_figure(data[2].cpu().numpy())
        # data2 = data[:, 16:] / data[:, :-16]
        # rff = torch.log(data2)
        # ts1 = torch.cat((sts1, lts1), dim=-1)
        # ts2 = torch.cat((sts2, lts2), dim=-1)

        rff1 = torch.log(torch.fft.fft(sts1)) - torch.log(torch.fft.fft(lts1))
        rff2 = torch.log(torch.fft.fft(sts2)) - torch.log(torch.fft.fft(lts2))
        rff = torch.cat((rff1, rff2, rff3), dim=-1)
        # rff = data_norm(rff)
        data1 = torch.cat((rff.real.unsqueeze(1).float(), rff.imag.unsqueeze(1).float()), dim=1)

        # ChannelIndSpectrogramObj = ChannelIndSpectrogram()
        # stft_rff = ChannelIndSpectrogramObj.channel_ind_spectrogram(data.cpu().numpy())
        # stft_rff = torch.from_numpy(stft_rff)
        # stft_rff = data_norm(stft_rff)
        # data_stft = torch.cat((stft_rff.real.unsqueeze(1).float(), stft_rff.imag.unsqueeze(1).float()), dim=1).to(device)
        return data1
    def process_rff(self,data):
        _,short, long, preamble = localSequenceGenerate()
        preamble = torch.from_numpy(preamble).unsqueeze(1).cuda()
        data_1 = data.resize(len(data), 20, 16)
        preamble = preamble.unsqueeze(0).expand(len(data), 320, 1).contiguous()
        preamble = preamble.resize(len(data), 20, 16)
        angle_i = torch.angle(data_1.mul(torch.conj(preamble)).sum(2))
        angle_i = torch.diff(angle_i, dim=1)

        sts1 = data[:, 16:16 * 5]
        sts2 = data[:, 16 * 5:16 * 9]
        lts1 = data[:, 192:192 + 64]
        lts2 = data[:, 192 + 64:192 + 64 * 2]
        rff1 = torch.log(torch.fft.fft(sts1)) - torch.log(torch.fft.fft(lts1))
        rff2 = torch.log(torch.fft.fft(sts2)) - torch.log(torch.fft.fft(lts2))
        # rff3 = torch.atan(torch.diff(data[:, 16 * 0:16 * 2]))-torch.atan(torch.diff(data[:, 160:192]))
        rff3 = torch.log(torch.fft.fft(data[:, 16 * 0:16 * 4])) - torch.log(torch.fft.fft(data[:, 160 - 32:160 + 32]))
        # rff = torch.cat((rff1, rff2), dim=-1)
        rff = torch.cat((rff1, rff2), dim=-1)
        # rff = data_norm(rff)
        data1 = torch.cat((rff.real.unsqueeze(1).float(), rff.imag.unsqueeze(1).float()), dim=1)
        return data1

    def process_rff3(self, data):

        batch_size = data.shape[0]
        sts1 = data[:, 32:16 * 6]
        sts2 = data[:, 16 * 6:16 * 10]
        lts1 = data[:, 192:192 + 64]
        lts2 = data[:, 192 + 64:192 + 64 * 2]

        plot_IQimage(np.array(sts1[0].cpu()))
        plot_IQimage(np.array(sts2[0].cpu()))
        # 生成本地序列 (NumPy 数组)
        short_16, short, long, preamble = localSequenceGenerate()

        # plot_IQimage(preamble)
        # 将理想序列转换为 PyTorch 张量 (与输入相同的设备和数据类型)
        device = data.device
        dtype = data.dtype

        long_tensor = torch.tensor(long, dtype=dtype, device=device)
        X_L1 = long_tensor[32:32 + 64]  # 第一个理想 LTS 符号
        X_L2 = long_tensor[32 + 64:32 + 128]  # 第二个理想 LTS 符号

        # 对理想 LTS 进行 FFT (频域表示)
        X_L1_freq = torch.fft.fft(X_L1)
        X_L2_freq = torch.fft.fft(X_L2)

        # 扩展理想频域序列以匹配批次大小
        X_L1_freq_batch = X_L1_freq.unsqueeze(0).expand(batch_size, -1)  # 形状 [batch_size, 64]
        X_L2_freq_batch = X_L2_freq.unsqueeze(0).expand(batch_size, -1)  # 形状 [batch_size, 64]

        # 对接收到的 LTS 进行 FFT
        Y_L1_freq = torch.fft.fft(lts1)  # 形状 [batch_size, 64]
        Y_L2_freq = torch.fft.fft(lts2)  # 形状 [batch_size, 64]

        # 使用 LS 估计计算信道响应 (避免除以零)
        # 创建掩码以避免除以零
        mask1 = (X_L1_freq_batch.abs() > 1e-6)  # 形状 [batch_size, 64]
        mask2 = (X_L2_freq_batch.abs() > 1e-6)  # 形状 [batch_size, 64]

        # 初始化结果张量
        H1 = torch.zeros_like(Y_L1_freq)  # 形状 [batch_size, 64]
        H2 = torch.zeros_like(Y_L2_freq)  # 形状 [batch_size, 64]

        # 仅在有意义的子载波上计算信道响应
        H1 = torch.where(mask1, Y_L1_freq / X_L1_freq_batch, H1)
        H2 = torch.where(mask2, Y_L2_freq / X_L2_freq_batch, H2)

        # 取两次估计的平均值作为最终信道响应
        H_avg = (H1 + H2) / 2  # 形状 [batch_size, 64]

        # 对两段STS分别进行FFT
        sts1_freq = torch.fft.fft(sts1)  # 形状 [batch_size, 64]
        sts2_freq = torch.fft.fft(sts2)  # 形状 [batch_size, 64]

        # 创建掩码避免除以零
        mask = (H_avg.abs() > 1e-6)

        # 初始化均衡后的频域信号
        eq_sts1_freq = torch.zeros_like(sts1_freq)
        eq_sts2_freq = torch.zeros_like(sts2_freq)

        # 信道均衡：除以信道响应
        eq_sts1_freq = torch.where(mask, sts1_freq / H_avg, eq_sts1_freq)
        eq_sts2_freq = torch.where(mask, sts2_freq / H_avg, eq_sts2_freq)

        # 应用IFFT转换回时域
        eq_sts1_time = torch.fft.ifft(eq_sts1_freq)  # 形状 [batch_size, 64]
        eq_sts2_time = torch.fft.ifft(eq_sts2_freq)  # 形状 [batch_size, 64]
        plot_IQimage(np.array(eq_sts1_time[0].cpu()))
        plot_IQimage(np.array(eq_sts2_time[0].cpu()))
        # 对两段均衡后的STS取平均
        z_spde_avg = (eq_sts1_time + eq_sts2_time) / 2  # 形状 [batch_size, 64]
        plot_IQimage(np.array(z_spde_avg[0].cpu()))
        data1 = torch.cat((z_spde_avg.real.unsqueeze(1).float(), z_spde_avg.imag.unsqueeze(1).float()), dim=1)
        return data1

    def train_base(self, train_loader, eval_loader):
        train_accba = AverageMeter()
        Loss =  AverageMeter()
        optimizer = optim.Adam([{'params': self.model.parameters()}], lr=0.001)
        self.model.train()
        best_acc = 0

        for j in range(self.args.max_epoch):
            for i, batch in enumerate(train_loader):
                x, seen_label = batch[0].cuda(), batch[1].cuda()

                # if x.dtype == torch.complex128:
                #     x = x.to(torch.complex64)
                # # 提取实部和虚部
                # real_part = x.real
                # imag_part = x.imag
                # # 沿新维度堆叠（通道维度）
                # x = torch.stack((real_part, imag_part), dim=1)

                x = self.freq_augmentation(x, 2000)
                x = self.phase_augmentation(x, 2)
                x = self.process_rff(x)
                logit = self.model(x)

                loss = F.cross_entropy(logit,seen_label)

                Loss.update(loss.item())
                acc = torch.eq(torch.argmax(logit, dim=1), seen_label).sum() /  seen_label.size(0)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_accba.update(acc.item())
            print("Epoch: : {:.1f}\t train_acc_ba (%): {:.5f}\t ".format(j,train_accba.avg))
            if j%3==0:

                with torch.no_grad():
                    test_acc = AverageMeter()
                    for i, batch in enumerate(eval_loader):
                        data_seen, seen_label = batch[0].cuda(), batch[1].cuda()

                        # if data_seen.dtype == torch.complex128:
                        #     data_seen = data_seen.to(torch.complex64)
                        # real_part = data_seen.real
                        # imag_part = data_seen.imag
                        # data_seen = torch.stack((real_part, imag_part), dim=1)

                        data_seen = self.process_rff(data_seen)
                        log_p_y = self.model(data_seen)

                        acc = torch.eq(torch.argmax(log_p_y, dim=1), seen_label).sum() / seen_label.size(0)
                        test_acc.update(acc.item())
                if test_acc.avg>best_acc:
                    best_acc =  test_acc.avg
                    weights = self.model.state_dict()
                    torch.save(weights, '/data1/hw/LYY/TTA/WiFi_TTA_log/WiFi_LTT.pth')
                print("test_acc_ba (%): {:.5f}\t ".format(test_acc.avg))
                print("best_acc_ba (%): {:.5f}\t ".format(best_acc))

    def enable_bn_adaptation(self,model):
        """启用BN层自适应"""
        model.train()
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm1d):
                m.track_running_stats = False

    def disable_bn_adaptation(self,model):
        """恢复默认测试模式"""
        model.eval()
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm1d):
                m.track_running_stats = True
    def train(self, train_loader, eval_loader):
        criterion = nn.CrossEntropyLoss().cuda()
        top_para = [v for k, v in self.model.named_parameters() if ('encoder' not in k and 'cls' not in k)]
        self.optimizer = optim.Adam([{'params': self.model.parameters()}],lr=0.0001)

        # self.optimizer = torch.optim.SGD([{'params': self.model.encoder.parameters(), 'lr': 0.0001},
        #                              {'params': top_para, 'lr': 0.01}],
        #                            momentum=0.9, nesterov=True, weight_decay=0.0005)

        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            self.args.max_epoch * len(train_loader),
            eta_min=0  # a tuning parameter
        )
        #self.model.load_state_dict(torch.load('/home/data/hw/LTT/Radio/FSOR-ADSB/log/model/ADS_B_LIMIT_aug.pth'))
        #pre_model_dict = self.train_base(50, self.train_gfsl_loader, self.model)
        #torch.save(pre_model_dict, '/home/data/hw/LTT/Radio/FSOR-ADSB/log/model/ADS_B_base.pth')
        self.model.load_state_dict(torch.load('/data1/hw/LYY/TTA/WiFi_TTA_log/WiFi_LTT.pth'))

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(eval_loader):
                data_seen, seen_label = batch[0].cuda(), batch[1].cuda()

                # if data_seen.dtype == torch.complex128:
                #     data_seen = data_seen.to(torch.complex64)
                # real_part = data_seen.real
                # imag_part = data_seen.imag
                # data_seen = torch.stack((real_part, imag_part), dim=1)

                data_seen = self.process_rff(data_seen)
                log_p_y = self.model(data_seen)

                acc = torch.eq(torch.argmax(log_p_y, dim=1), seen_label).sum() / seen_label.size(0)
        log_str_init = 'Inital Accuracy = {}'.format(acc)
        print(log_str_init + '\n')
        for epoch in range(0, self.args.max_epoch + 1):
            #
            Theta = self.theta_cal(train_loader)
            print(Theta)
            # self.model.train()
            # #
            # train_acc,train_loss = self.train_episode(epoch, train_loader,Theta,self.args)
            # # #self.replace_base_fc(self.train_gfsl_loader, self.args)
            # print( "==> Epoch {}/{}".format(epoch + 1, self.args.max_epoch))
            # print(
            #     "ACC (%): {:.5f}\t  loss1 (%): {:.5f}\t "
            #         .format(train_acc, train_loss))
            # torch.save( self.model.state_dict(), '/data1/hw/LTT/WiFi_TTA/log/WiFi_v2.pth')
            if (epoch) % 5 == 0:
                #self.model.load_state_dict(torch.load('/data1/hw/LTT/WiFi_TTA/log/WiFi_v1.pth'))
                self.model.eval()
                self.test(eval_loader, Theta)

                #theta = self.theta_cal()
                #self.Neg_pro.load_state_dict(torch.load('/home/data/hw/LTT/Radio/FSOR-ADSB/log/model/ADS_B_cross_v1.pth'))
                #self.theta_cal4()
                #weights = self.model.state_dict()
                #torch.save(weights, '/home/data/hw/LTT/Radio/FSOR-ADSB/log/model/ADS_B_aug_v1.pth')

                #self.model.load_state_dict(torch.load('/home/data/hw/LTT/Radio/FSOR-ADSB/log/model/ADS_B.pth'))

                #self.test(eval_loader,theta)

                #self.test4(eval_loader, theta)

        print(log_str_init + '\n')

    def theta_cal(self,train_loader):
        self.model.eval()
        logits,Y=[],[]
        with torch.no_grad():
            for step,(data,labels) in enumerate(train_loader):
                X1 = data.cuda()
                y = labels.cuda()
                with torch.set_grad_enabled(False):
                    logits_s= self.model.forward_test(X1)
                    logits.append(logits_s)
                    Y.append(y)
            logits = torch.cat(logits, dim=0)
            Y = torch.cat(Y, dim=0)
            acc = torch.eq(torch.argmax(logits, dim=1), Y).sum() / Y.size(0)
            print('G_acc: {:.5f}'.format(acc))
        scores = [[] for _ in range(10)]
        for score,  t in zip(logits,Y):
            if torch.argmax(score) == t:
                scores[t].append(torch.max(score).unsqueeze(dim=0))
        Theta = []
        for i in range(10):
            RE = scores[i]
            if RE != []:
                RE = torch.cat(RE, 0).cpu().numpy()
            else:
                RE = torch.Tensor([0])
            new_RE = np.sort(RE)
            index = len(new_RE)-int(len(new_RE)*0.99)
            theta = new_RE[index]
            # for j in range(len(new_RE)):
            #     k = (RE > new_RE[j]).sum()
            #     if k / len(new_RE) <= 0.95:
            #         theta = new_RE[j]
            #         break
            Theta.append(theta)

        return Theta


    def test(self,eval_loader,theta):
        cls_ori = deepcopy(self.model.cls.weight.data)
        eval_loader2 = eval_loader

        with torch.no_grad():
            #test_accba = AverageMeter()
            for i, batch in enumerate(eval_loader):
                #self.enable_bn_adaptation(self.model)
                data_seen, seen_label = batch[0].cuda(), batch[1].cuda()

                # if data_seen.dtype == torch.complex128:
                #     data_seen = data_seen.to(torch.complex64)
                # real_part = data_seen.real
                # imag_part = data_seen.imag
                # data_seen = torch.stack((real_part, imag_part), dim=1)

                x_aug = self.freq_augmentation(data_seen, 2000)
                x_aug = self.phase_augmentation(x_aug, 2)
                x_aug = self.process_rff(x_aug)
                data_seen_ = self.process_rff(data_seen)

                if i ==0:
                    cls=cls_ori
                logit, cls = self.model._forward_gfsl_test(data_seen_,x_aug, seen_label, theta, cls, i)

                #accba = torch.eq(torch.argmax(logit, dim=1), seen_label).sum() / seen_label.size(0)
                #test_accba.update(accba.item())
                #self.disable_bn_adaptation(self.model)

                test_accba = AverageMeter()
                for j, batch in enumerate(eval_loader2):
                    data_seen, seen_label = batch[0].cuda(), batch[1].cuda()

                    # if data_seen.dtype == torch.complex128:
                    #     data_seen = data_seen.to(torch.complex64)
                    # real_part = data_seen.real
                    # imag_part = data_seen.imag
                    # data_seen = torch.stack((real_part, imag_part), dim=1)

                    data_seen = self.process_rff(data_seen)
                    logit= self.model._forward_test(data_seen, cls, cls_ori)
                    accba = torch.eq(torch.argmax(logit, dim=1), seen_label).sum() / seen_label.size(0)
                    test_accba.update(accba.item())
                print('epoch: {:.5f}\ttest_acc:{:.5f}\t'.format(i, test_accba.avg))


    def train_episode(self, epoch, train_gfsl_loader,Theta,args):
        accfs = AverageMeter()
        accmn = AverageMeter()
        losses_cls = AverageMeter()
        losses_funit = AverageMeter()
        train_step = 0
        tqdm_gen = tqdm(train_gfsl_loader)

        for step, batch in enumerate(tqdm_gen,1):
            train_step += 1
            x, support_label = batch[0].cuda(), batch[1].cuda()

            logits, mask_ = self.model.forward_(x, support_label,Theta, args)
            #loss_r = mask_.mean()
            label = support_label.view(1, -1).repeat(args.num_tasks, 1).view(-1)
            loss = F.cross_entropy(logits,label)
            acc = count_acc(logits, label)

            # ===================backward=====================
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            accfs.update(acc)
            losses_funit.update(loss.item())


        return accfs.avg,  losses_funit.avg

