import torch.nn as nn
import torch
import torch.nn.functional as F
from resnet_ import ResNet18
from util import  AverageMeter,one_hot
import numpy as np
from sklearn import metrics
import math
import random
from utils import *
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn, log_attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        # combine local and global key and value
        k = torch.cat([q, k], 1)
        v = torch.cat([q, v], 1)
        len_k = len_k + len_q
        len_v = len_v + len_q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        output, attn, log_attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = output + residual
        return output, attn, log_attn

def sample_task_ids(support_label, num_task, num_shot, num_way, num_class):
    basis_matrix = torch.arange(num_shot).long().view(-1, 1).repeat(1, num_way).view(-1) * num_class
    permuted_ids = torch.zeros(num_task, num_shot * num_way).long()
    permuted_labels = []
    for i in range(num_task):
        clsmap = torch.randperm(num_class)[:num_way]
        permuted_labels.append(support_label[clsmap])
        permuted_ids[i, :].copy_(basis_matrix + clsmap.repeat(num_shot))

    return permuted_ids, permuted_labels


class FeatureNet(nn.Module):
    def __init__(self,args):
        super(FeatureNet,self).__init__()
        self.encoder =ResNet18(512)
        self.slf_attn = MultiHeadAttention(1, 512,512, 512, dropout=0.1)
        self.cls = nn.Linear(512,10, bias=True)
        #self.encoder = RNN(input_size=1024, hidden_size=1024, num_layers=2, num_classes=512)
        #self.shared_key = torch.nn.Parameter(torch.nn.init.kaiming_normal_(torch.zeros(1, 237 * 2, 512)))
    def forward(self, x):
        f = self.encoder(x)
        f = self.cls(f)
        # current_classifier = F.normalize(self.cls.weight, dim=-1)
        # sim_marix = torch.mm(current_classifier, current_classifier.t())  # 64,437,512
        # mask = torch.ones(sim_marix.size()) - torch.eye(sim_marix.size(0))
        # mask_sim = torch.mul(mask.cuda(), sim_marix)
        return f

    def forward_test(self, x):
        x1 = self.process_rff(x)

        # if x.dtype == torch.complex128:
        #     x = x.to(torch.complex64)
        # # 提取实部和虚部
        # real_part = x.real
        # imag_part = x.imag
        # # 沿新维度堆叠（通道维度）
        # x1 = torch.stack((real_part, imag_part), dim=1)

        f = self.encoder(x1)

        current_classifier = F.normalize(self.cls.weight.t(), dim=-1)
        x_aug_f = F.normalize(f, dim=-1)

        logits_psi = torch.mm(x_aug_f, current_classifier)
        return logits_psi,f

    def compute_prototypes(self,logits_psi, support_x, support_y,Theta,cls):

        # 获取所有唯一类别
        pre_label = torch.argmax(logits_psi, dim=1)
        classes = [i for i in range(10)]
        H_ens=-torch.sum(F.softmax(logits_psi) * torch.log(F.softmax(logits_psi)), dim=1)
        #H_ens = -F.softmax(logits_psi).unsqueeze(1).bmm(torch.log(F.softmax(logits_psi)).unsqueeze(1).permute(0,2,1))

        #logits_max = torch.max(logits_psi,dim=1)[0]

        # 初始化存储原型点的列表
        prototypes = cls
        flag=0
        k=5

        for c in classes:
            # 筛选当前类别的所有样本特征
            mask = (pre_label == c)
            if len(mask) != 0:
                H_ens_c = H_ens[mask]
                class_features1 = support_x[mask]
                if len(class_features1)>=k:
                    top5_min_indices = torch.topk(H_ens_c.flatten(), k=k, largest=False).indices
                    class_features1 = class_features1[top5_min_indices]
                    prototype = class_features1.mean(dim=0)

                    prototypes[c] = prototype
                    flag = 1
                # else:
                #     top5_min_indices = torch.topk(H_ens_c.flatten(), k= len(class_features1), largest=False).indices
                #     class_features1 = class_features1[top5_min_indices]

            # if len(mask)!=0:
            #     mask_in = torch.where(logits_max[mask] > Theta[c])[0]
            #     if len(mask_in)!=0:
            #         class_features1 = support_x[mask]
            #         class_features1 = class_features1[mask_in]
            #         # 计算当前类别的原型（均值）
            #         prototype = class_features1.mean(dim=0)
            #         prototypes[c]=prototype
            #         flag=1
        # 将所有原型点堆叠成一个张量
        #prototypes = torch.stack(prototypes)

        return prototypes,flag

    def forward_(self,  x, support_label,Theta, args):
        # x1 = self.process_rff(x)
        # x_f = self.encoder(x1)
        # x_f = F.normalize(x_f, dim=-1)
        logit,mask_ = [],[]
        for tt in range( args.num_tasks):

            x_aug = self.freq_augmentation(x, 2000)
            x_aug = self.phase_augmentation(x_aug, 2)
            x_aug = self.process_rff(x_aug)
            x_aug_f = self.encoder(x_aug)
            x_aug_f = F.normalize(x_aug_f, dim=-1)
            current_classifier = F.normalize(self.cls.weight.t(), dim=-1)
            logits_psi = torch.mm(x_aug_f, current_classifier)

            prototypes,flag = self.compute_prototypes(logits_psi,x_aug_f, support_label,Theta) #10,512
            cls = 0.9 *self.cls.weight.data + (0.1) * prototypes
            current_classifier = cls.unsqueeze(0).expand(x_aug_f.size(0),10,512).contiguous()
            current_classifier = torch.cat([current_classifier, x_aug_f.unsqueeze(1)], 1)

            combined, _, _ = self.slf_attn(current_classifier, current_classifier, current_classifier)
            current_classifier, data = combined.split(10, 1)
            #current_classifier = combined.squeeze(0)
            current_classifier = F.normalize(current_classifier, dim=-1)

            query = data
            logit.append(F.cosine_similarity(query, current_classifier,dim=-1))
            # sim_marix = torch.mm(current_classifier, current_classifier.t())  # 64,437,512
            # mask = torch.ones(sim_marix.size()) - torch.eye(sim_marix.size(0))
            # mask_sim = torch.mul(mask.cuda(), sim_marix)
            # mask_.append(abs(mask_sim.unsqueeze(0)))
        #mask_ = torch.cat(mask_, 0)
        logit = torch.cat(logit, 0).squeeze(1)
        return logit, mask_

    def _forward_gfsl_test(self, data_seen,x_aug, seen_label,Theta,cls,step):
        f_qry = self.encoder(data_seen)
        f_qry = F.normalize(f_qry, dim=-1)

        f_aug = self.encoder(x_aug)
        f_aug = F.normalize(f_aug, dim=-1)
        f_ens = (f_qry+f_aug)/2

        current_classifier = F.normalize(cls.t(), dim=-1)
        logits_1 = torch.mm(f_qry, current_classifier)
        logits_2 = torch.mm(f_aug, current_classifier)
        logits_ens = (logits_1+logits_2)/2

        prototypes_ens, flag = self.compute_prototypes(logits_ens, f_ens, seen_label, Theta, cls)

        H_ens = -torch.sum(F.softmax(logits_ens) * torch.log(F.softmax(logits_ens)), dim=1)

        alpha_base = 0.9 * (0.995 ** step)
        alpha = alpha_base * (H_ens.mean() ** 0.5)
        cls1 = alpha * cls + (1 - alpha) * prototypes_ens


        logits_pro = torch.mm(f_ens, cls1.t())
        H_pro = -torch.sum(F.softmax(logits_pro) * torch.log(F.softmax(logits_pro)), dim=1)

        mask = H_ens < H_pro

        # 合并结果 (B, 10)
        combined_probs = torch.where(mask.unsqueeze(1), logits_ens, logits_pro)

        # current_classifier = F.normalize(cls.t(), dim=-1)
        # logits_psi = torch.mm(f_qry, current_classifier)
        # prototypes, flag = self.compute_prototypes(logits_psi, f_qry, seen_label, Theta,cls)  # 10,512

        # 置信度调整
        #alpha = alpha_base * (torch.max(F.softmax(logits_psi),dim=1)[0].mean() ** 0.5)
        #cls1 = alpha*cls+ (1-alpha)*prototypes

        #combined, _, _ = self.slf_attn(prototypes.unsqueeze(0), cls.unsqueeze(0),
        #                                 cls.unsqueeze(0))
        #current_classifier = combined.squeeze(0)
        #logits_s = torch.mm(f_qry, cls.t())
        return logits_pro,cls1

    def _forward_test(self, data_seen, current_classifier,cls_ori):
        f_qry = self.encoder(data_seen)
        f_qry = F.normalize(f_qry, dim=-1)
        current_classifier = F.normalize(current_classifier, dim=-1)

        #可修改
        logits_pro = torch.mm(f_qry, current_classifier.t())

        current_classifier2 = F.normalize(cls_ori, dim=-1)
        logits_ens = torch.mm(f_qry, current_classifier2.t())

        H_ens = -torch.sum(F.softmax(logits_ens) * torch.log(F.softmax(logits_ens)), dim=1)
        H_pro = -torch.sum(F.softmax(logits_pro) * torch.log(F.softmax(logits_pro)), dim=1)
        mask = H_ens < H_pro

        # 合并结果 (B, 10)
        combined_probs = torch.where(mask.unsqueeze(1), logits_ens, logits_pro)
        return logits_pro

    def EuclideanDistances(self,a, b):
        sq_a = a ** 2
        sum_sq_a = torch.sum(sq_a, dim=1).unsqueeze(1)  # m->[m, 1]
        sq_b = b ** 2
        sum_sq_b = torch.sum(sq_b, dim=1).unsqueeze(0)  # n->[1, n]
        bt = b.t()
        return torch.sqrt(sum_sq_a + sum_sq_b - 2 * a.mm(bt))

    def update(self, data, label, class_list,cls_weight):
        #data,label = self.augmentation(data, label)
        _,data = self.encoder(data)
        data=data.detach()
        for class_index in class_list:
            # print(class_index)
            data_index = (label == class_index).nonzero().squeeze(-1)
            embedding = data[data_index]
            proto = embedding.mean(0)
            cls_weight.data[class_index] = proto
            # print(proto)
        cls=cls_weight.unsqueeze(0)
        pro_,_,_ = self.slf_attn(cls, cls, cls)
        cls = F.normalize(pro_.squeeze(0), dim=-1)
        logits = torch.mm(data, cls.t()) / 1

        return logits,label, cls_weight


    def freq_augmentation(self, data, freq):
        # cfo_det1 = (torch.rand(data.size(0), 1)-0.5)*2*2000
        cfo_det = (torch.rand(data.size(0), 1) * freq).cuda()
        # noise = (torch.rand(data.size(0), 1)* 0.1).to(device)
        # raw_data1 = torch.complex(data[:, 0, :], data[:, 1, :]).squeeze(-1)
        # raw_data2 = torch.complex(data[:, 2, :], data[:, 3, :]).squeeze(-1)
        t = torch.arange(320).cuda() / 20e6
        data1 = data * torch.exp(1j * 2 * torch.pi * cfo_det * t)
        # data2 = raw_data2 * torch.exp(1j * 2 * torch.pi * cfo_det * t)
        return data1

    def phase_augmentation(self, data, phase):
        phase_det = (torch.rand(data.size(0), 1) * phase).cuda()
        data1 = data * torch.exp(1j * (-phase_det))
        return data1

    def process_rff1(self, data):
        sts1 = data[:, 16:16 * 5]
        sts2 = data[:, 16 * 5:16 * 9]
        lts1 = data[:, 192:192 + 64]
        lts2 = data[:, 192 + 64:192 + 64 * 2]
        # sts = torch.cat((sts1, sts2), dim=-1)
        # lts = torch.cat((lts1, lts2), dim=-1)
        # sts0 = data[:, :16*4], data[:, 160-32:160+32]
        rff3 = torch.log(torch.fft.fft(data[:, 16 * 0:16 * 4])) - torch.log(
            torch.fft.fft(data[:, 160 - 32:160 + 32]))
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

    def process_rff(self, data):
        _, short, long, preamble = localSequenceGenerate()
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
        rff3 = torch.log(torch.fft.fft(data[:, 16 * 0:16 * 4])) - torch.log(
            torch.fft.fft(data[:, 160 - 32:160 + 32]))
        # rff = torch.cat((rff1, rff2), dim=-1)
        rff = torch.cat((rff1, rff2), dim=-1)
        # rff = data_norm(rff)
        data1 = torch.cat((rff.real.unsqueeze(1).float(), rff.imag.unsqueeze(1).float()), dim=1)
        return data1




    def forward_out(self,data):
        _,f = self.encoder(data)
        return f.detach()

class Neg_pro(nn.Module):
    def __init__(self):
        super(Neg_pro, self).__init__()
        self.attn1 = MultiHeadAttention(1, 512, 512, 512, dropout=0.1)
        self.mlp1 = nn.Linear(512,512)

        self.mlp2 = nn.Linear(512, 512)

    def forward(self, cls, labels):
        num_batch = labels.size(0)
        pro = cls[labels]

        cls = cls.unsqueeze(0).expand(num_batch, 437, 512).contiguous()
        pro_ = pro.unsqueeze(1).expand(num_batch, 437, 512).contiguous()
        p_, _, _ = self.attn1(cls,pro_,cls) # 256,437,512
        idx = (1-one_hot(labels, 437)).unsqueeze(2).expand(num_batch,437,512) # 256,437
        p_ = p_.mul(idx)

        p_ = p_.mean(1)
        p_ = F.leaky_relu(self.mlp1(p_))
        neg = self.mlp2(F.leaky_relu(pro.mul(p_)))
        return pro,neg


class CrossNet(nn.Module):
    def __init__(self):
        super(CrossNet,self).__init__()
        self.attn = MultiHeadAttention(1,512,512,512, dropout=0.1)
        self.fc1 = nn.Linear(1024,512)
        self.fc2 = nn.Linear(512,2)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x,logits_):
        #x = torch.cat((x, logits_.unsqueeze(1)), dim=2)
        #x = x.view(x.size(0),1, -1)
        #x_1 = x[:,0,:].mul(x[:,1,:]).unsqueeze(1)
        #x = torch.cat((x,x_1),dim=1)
        #x = x.unsqueeze(1)
        x, _, _ = self.attn(x,x,x)
        x = x.view(x.size(0),-1)

        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return F.sigmoid(x)

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, X, W_norm):
        out = torch.matmul(W_norm, X)
        out = self.linear(out)
        return out

class WGCN(nn.Module):
    def __init__(self, in_features, hidden_features, num_classes):
        super(WGCN, self).__init__()
        self.gc1 = GraphConvolution(in_features, hidden_features)
        self.gc1_ = GraphConvolution(hidden_features, 64)
        self.gc2 = GraphConvolution(in_features, hidden_features)
        self.gc2_ = GraphConvolution(hidden_features,  64)
        self.fc = nn.Linear(64*437, num_classes)
        self.attn = MultiHeadAttention(1,64, 64, 64, dropout=0.1)
    def forward(self, X,X_n):
        out1 = self.encoder(X)
        out2 = self.encoder(X_n)

        return torch.cat((out1,out2),dim=0)

    def encoder(self,X):
        X1, X2 = X[0: 437, :], X[437:, :]
        W1_norm, W2_norm = torch.mm(X1, X1.t()), torch.mm(X2, X2.t())

        X1 = F.leaky_relu(self.gc1(X1, W1_norm))
        X1 = F.leaky_relu(self.gc1_(X1, W1_norm))

        X2 = F.leaky_relu(self.gc2(X2, W2_norm))
        X2 = F.leaky_relu(self.gc2_(X2, W2_norm))

        # X = torch.cat((X1,X2),dim=)
        X, _, _ = self.attn(X1.unsqueeze(0), X2.unsqueeze(0), X2.unsqueeze(0))
        X = X.view(-1, 64 * 437)
        output = F.sigmoid(self.fc(X))
        return output