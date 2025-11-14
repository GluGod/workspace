import os
import argparse
from torch.utils.data import DataLoader
import torch
from dataset import  Wifi
#from samplers import CategoriesSampler, RandomSampler
from MetaTrainer import MetaTrainer
import numpy as np
import torch.utils.data as Data
import warnings
from util import *
warnings.filterwarnings("ignore")

root='/home/data/hw/LTT/Radio/'
parser = argparse.ArgumentParser("ADS_B")

# Dataset
parser.add_argument('--dataset', type=str, default='WiFi', help="WiFi")

parser.add_argument('--dataroot', type=str, default=root)

# Meta Setting
parser.add_argument('--n_ways', type=int, default=5, metavar='N', help='Number of classes for doing each classification run')
parser.add_argument('--n_shots', type=int, default=1, metavar='N', help='Number of shots in test')
parser.add_argument('--n_queries', type=int, default=15, metavar='N', help='Number of query in test')
parser.add_argument('--n_open_ways', type=int, default=5, metavar='N', help='Number of classes for doing each classification run')
parser.add_argument('--n_train_runs', type=int, default=300, help='Number of training episodes')
parser.add_argument('--n_test_runs', type=int, default=600, metavar='N', help='Number of test runs')

# optimization
parser.add_argument('--sample_class', type=int, default=64)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_tasks', type=int, default=1)
parser.add_argument('--max_epoch', type=int, default=100)
parser.add_argument('--seed', type=int, default=22)
parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
#
parser.add_argument('--num_workers', type=int, default=16)
parser.add_argument('--gpus', type=str, default='0')

args = parser.parse_args()
print('begin_train')

set_seed(41)

data_1 =  Wifi(args, num_workers=16, batch_size=args.batch_size)
train_dataloader = data_1.train_loader
test_dataloader = data_1.test_loader

trainer = MetaTrainer(args)
trainer.train_base(train_dataloader,test_dataloader)
trainer.train(train_dataloader,test_dataloader)

# data = np.load(root + 'Task_ADS-B/train_x_100_aug.npy')
# labels = np.load(root + 'Task_ADS-B/train_y_100.npy')
# train_dataset = Data.TensorDataset(torch.Tensor(data ), torch.LongTensor(labels))
# train_gfsl_loader = DataLoader(dataset=train_dataset,
#                                    batch_size=args.batch_size,
#                                    shuffle=True,
#                                    num_workers=16,
#                                    pin_memory=True)
#
# train_sampler = CategoriesSampler(torch.LongTensor(labels),
#                                       len(train_gfsl_loader),
#                                       args.sample_class,
#                                       args.n_shots)
#
# train_fsl_loader = DataLoader(dataset=train_dataset,
#                                    batch_sampler=train_sampler,
#                                    num_workers=16,
#                                    pin_memory=True)
#
# data = np.load(root + 'Task_ADS-B/test_x_100_aug.npy')
# labels = np.load(root + 'Task_ADS-B/test_y_100.npy')
# test_base_dataset = Data.TensorDataset(torch.Tensor(data), torch.LongTensor(labels))
# test_gfsl_loader = DataLoader(dataset=test_base_dataset,
#                                    batch_size=args.batch_size,
#                                    shuffle=True,
#                                    num_workers=16,
#                                    pin_memory=True)
#
# test_fsl_loader = fsl_test_data(args)

'''
test_fsl_dataset = Data.TensorDataset(torch.Tensor(data), torch.LongTensor(labels))
val_sampler = CategoriesSampler(torch.LongTensor(labels),
                                    500,
                                    5,
                                    1 + 15)
test_fsl_loader = DataLoader(dataset=test_fsl_dataset,
                                   batch_sampler=val_sampler,
                                   num_workers=16,
                                   pin_memory=True)
'''


# data = np.load(root + 'Task_ADS-B/test_x_10_aug.npy')
# labels = np.load(root + 'Task_ADS-B/test_y_10.npy')
# test_out_dataset = Data.TensorDataset(torch.Tensor(data), torch.LongTensor(labels))
# test_out_loader = DataLoader(dataset=test_out_dataset,
#                                    batch_size=args.batch_size,
#                                    shuffle=True,
#                                    num_workers=16,
#                                    pin_memory=True)





