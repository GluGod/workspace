import os
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
        preamble = preamble.unsqueeze(0).expand(len(data), 320,1).contiguous()
        preamble = preamble.squeeze(2)
        preamble = preamble.resize(len(data), 20, 16)
        angle_i = torch.angle(data_1.mul(torch.conj(preamble)).sum(2))
        angle_i = torch.diff(angle_i, dim=1)
        #data =data-preamble


        sts1 = data[:, 16:16 * 5]
        sts2 = data[:, 16 * 5:16 * 9]
        lts1 = data[:, 192:192 + 64]
        lts2 = data[:, 192 + 64:192 + 64 * 2]
        rff1 = torch.log(torch.fft.fft(sts1)) - torch.log(torch.fft.fft(lts1))
        rff2 = torch.log(torch.fft.fft(sts2)) - torch.log(torch.fft.fft(lts2))
        # rff3 = torch.atan(torch.diff(data[:, 16 * 0:16 * 2]))-torch.atan(torch.diff(data[:, 160:192]))
        rff3 = torch.log(torch.fft.fft(data[:, 16 * 0:16 * 4])) - torch.log(torch.fft.fft(data[:, 160 - 32:160 + 32]))
        # rff = torch.cat((rff1, rff2), dim=-1)
        rff = torch.cat((rff1,rff2), dim=-1)
        # rff = data_norm(rff)
        # plt.figure()
        # plt.plot(rff[0].real.cpu().numpy())
        # plt.savefig('/data1/hw/LTT/WiFi_TTA/log/kl.jpg')

        data1 = torch.cat((rff.real.unsqueeze(1).float(), rff.imag.unsqueeze(1).float()), dim=1)
        return data1

    def data_norm(self,data_tensor):
        data_norm = torch.zeros_like(data_tensor, dtype=torch.cfloat)
        for i in range(data_tensor.shape[0]):
            sig_amplitude = torch.abs(data_tensor[i])
            rms = torch.sqrt(torch.mean(sig_amplitude ** 2))
            data_norm[i] = data_tensor[i] / rms
        return data_norm
    def process_rff2(self,data):
        _,short, long, preamble = localSequenceGenerate()
        preamble = torch.from_numpy(preamble).unsqueeze(1).cuda()
        data_1 = data.resize(len(data), 20, 16)
        preamble = preamble.unsqueeze(0).expand(len(data), 320,1).contiguous()
        preamble = preamble.squeeze(2)

        sts1 = data[:, 16:16 * 5]
        sts2 = data[:, 16 * 10:16 *14]
        lts1 = data[:, 192:192 + 64]
        lts2 = data[:, 0:64]
        rff1 = torch.log(torch.fft.fft(sts1)) - torch.log(torch.fft.fft(sts2))
        rff2 = torch.log(torch.fft.fft(lts1)) - torch.log(torch.fft.fft(lts2))
        #rff3 = torch.log(torch.fft.fft(lts1)) - torch.log(torch.fft.fft(lts1)).mean(1).unsqueeze(1)
        #rff4 = torch.log(torch.fft.fft(lts2)) - torch.log(torch.fft.fft(lts2)).mean(1).unsqueeze(1)
        # plt.figure()
        # plt.plot(rff3[0].real.cpu().numpy())
        # plt.savefig('/data1/hw/LTT/WiFi_TTA/log/k1.jpg')
        # rff4 = torch.log(torch.fft.fft(lts1)) - torch.log(torch.fft.fft(lts2))
        #rff3 = torch.atan(torch.diff(data[:, 16 * 0:16 * 2]))-torch.atan(torch.diff(data[:, 160:192]))
        #rff3 = torch.log(torch.fft.fft(data[:, 16 * 0:16 * 4])) - torch.log(torch.fft.fft(data[:, 160 - 32:160 + 32]))
        # rff = torch.cat((rff1, rff2), dim=-1)
        rff = torch.cat((rff1,rff2), dim=-1)
        # rff = data_norm(rff)
        data1 = torch.cat((rff.real.unsqueeze(1).float(), rff.imag.unsqueeze(1).float()), dim=1)

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
                        data_seen = self.process_rff(data_seen)
                        log_p_y = self.model(data_seen)

                        acc = torch.eq(torch.argmax(log_p_y, dim=1), seen_label).sum() / seen_label.size(0)
                        test_acc.update(acc.item())
                if test_acc.avg>best_acc:
                    best_acc =  test_acc.avg
                    weights = self.model.state_dict()
                    torch.save(weights, '/data1/hw/LTT/WiFi_TTA/log/WiFi_v3.pth')
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
        self.model.load_state_dict(torch.load('/data1/hw/LTT/WiFi_TTA/log/WiFi_v3.pth'))

        for epoch in range(0, self.args.max_epoch + 1):

            F, y = self.theta_cal(train_loader)
            print(F.size(),y.size())

            from sklearn.neighbors import KNeighborsClassifier

            # 创建KNN分类器实例
            knn = KNeighborsClassifier(
                n_neighbors=5,  # 选择K值(邻居数量)
                weights='uniform',  # 'uniform'或'distance'(距离加权)
                algorithm='auto',  # 计算最近邻的算法
                p=2  # 距离度量(1:曼哈顿距离，2:欧氏距离)
            )

            # 训练模型
            knn.fit(F.cpu().numpy(), y.cpu().numpy())
            # torch.save( self.model.state_dict(), '/data1/hw/LTT/WiFi_TTA/log/WiFi_v2.pth')
            if (epoch) % 5 == 0:
                #self.model.load_state_dict(torch.load('/data1/hw/LTT/WiFi_TTA/log/WiFi_v1.pth'))
                self.model.eval()
                self.test(eval_loader, knn,F.cpu().numpy(), y.cpu().numpy())

    def theta_cal(self,train_loader):
        self.model.eval()
        logits,Y=[],[]
        F=[]
        with torch.no_grad():
            for step,(data,labels) in enumerate(train_loader):
                X1 = data.cuda()
                y = labels.cuda()
                with torch.set_grad_enabled(False):
                    logits_s,f= self.model.forward_test(X1)
                    logits.append(logits_s)
                    Y.append(y)
                    F.append(f)
            logits = torch.cat(logits, dim=0)
            Y = torch.cat(Y, dim=0)
            F=torch.cat(F,dim=0)

        return F, torch.argmax(logits, dim=1)


    def test(self,eval_loader,knn, F, Y):
        cls_ori = deepcopy(self.model.cls.weight.data)
        eval_loader2 = eval_loader
        from sklearn.metrics import accuracy_score
        with torch.no_grad():
            #test_accba = AverageMeter()
            for i, batch in enumerate(eval_loader):
                #self.enable_bn_adaptation(self.model)
                data_seen, seen_label = batch[0].cuda(), batch[1].cuda()
                # x_aug = self.freq_augmentation(data_seen, 2000)
                # x_aug = self.phase_augmentation(x_aug, 2)
                # x_aug = self.process_rff(x_aug)
                logits_s, f = self.model.forward_test(data_seen)
                y = knn.predict(f.cpu().numpy())

                F = np.concatenate((F, f.cpu().numpy()),axis=0)
                Y = np.concatenate((Y,  y),axis=0)
                knn.fit(F, Y)

                # if i ==0:
                #     cls=cls_ori
                # logit, cls,prototypes_ens = self.model._forward_gfsl_test(data_seen_,x_aug, seen_label, theta, cls, i)
                #accba = torch.eq(torch.argmax(logit, dim=1), seen_label).sum() / seen_label.size(0)
                #test_accba.update(accba.item())
                #self.disable_bn_adaptation(self.model)

                test_accba = AverageMeter()
                for j, batch in enumerate(eval_loader2):
                    data_seen, seen_label = batch[0].cuda(), batch[1].cuda()
                    logits_s, f = self.model.forward_test(data_seen)
                    y = knn.predict(f.cpu().numpy())
                    #logit= self.model._forward_test(data_seen, cls, prototypes_ens)
                    accba = accuracy_score(y, seen_label.cpu().numpy())
                    #accba = torch.eq(torch.argmax(logit, dim=1), seen_label).sum() / seen_label.size(0)
                    test_accba.update(accba)
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

