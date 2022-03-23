from __future__ import print_function
import argparse
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Net
from data import get_training_set, get_test_set

from matplotlib import pyplot as plt
import torch.nn.functional as F
import datetime

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

starttime = datetime.datetime.now()

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, required=True, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=64, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=10, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=2, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
opt = parser.parse_args()

print(opt)

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)

device = torch.device("cuda" if opt.cuda else "cpu")

print('===> Loading datasets')
train_set = get_training_set(opt.upscale_factor)
test_set = get_test_set(opt.upscale_factor)
# training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
# testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)
training_data_loader = DataLoader(dataset=train_set, batch_size=opt.batchSize, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, batch_size=opt.testBatchSize, shuffle=False)

print('===> Building MyModel')
model = Net(upscale_factor=opt.upscale_factor).to(device)
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=opt.lr)

loss_x = []
loss_y = []
psnr_x = []
psnr_y = []


def train(epoch):
    epoch_loss = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = batch[0].to(device), batch[1].to(device)

        result = model(input)
        optimizer.zero_grad()
        loss = criterion(result, target)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.item()))

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))
    loss_x.append(epoch)
    loss_y.append(epoch_loss / len(training_data_loader))


def test():
    avg_psnr = 0
    with torch.no_grad():
        for batch in testing_data_loader:
            input, target = batch[0].to(device), batch[1].to(device)

            prediction = model(input)
            mse = criterion(prediction, target)
            psnr = 10 * log10(1 / mse.item())
            avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))
    psnr_y.append(avg_psnr / len(testing_data_loader))


def train_cascade(epoch):
    epoch_loss = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = batch[0].to(device), batch[1].to(device)
        # input：torch.Size([4, 1, 85, 85])

        # input_min = F.interpolate(input, size=[80, 80], scale_factor=0.5)
        input_min = F.interpolate(input, size=[60, 60])
        input_min_min = F.interpolate(input_min, size=[40, 40])
        # input_min_min_min = F.interpolate(input_min_min, size=[70, 70])

        result = model(input)
        # print(result.size())
        result_min = model(input_min)
        # print(result_min.size())
        result_min_min = model(input_min_min)
        # print(result_min_min.size())
        # result_min_min_min = model(input_min_min_min)

        result2 = F.interpolate(result_min, size=[255,255])
        result3 = F.interpolate(result_min_min, size=[255,255])
        # result4 = F.interpolate(result_min_min_min, size=[255, 255])

        result_all = result + result2 + result3
        # result_all = result
        # target：torch.Size([4, 1, 255, 255])
        optimizer.zero_grad()
        loss = criterion(result_all, target)

        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.item()))

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))
    loss_x.append(epoch)
    loss_y.append(epoch_loss / len(training_data_loader))

def test_cascade():
    avg_psnr = 0
    with torch.no_grad():
        for batch in testing_data_loader:
            input, target = batch[0].to(device), batch[1].to(device)
            # ---------------------
            input_min = F.interpolate(input, size=[60,60])
            input_min_min = F.interpolate(input_min, size=[40,40])
            # input_min_min_min = F.interpolate(input_min_min, size=[70,70])

            result = model(input)
            # print(result.size())
            result_min = model(input_min)
            # print(result_min.size())
            result_min_min = model(input_min_min)
            # print(result_min_min.size())
            # result_min_min_min = model(input_min_min_min)

            result2 = F.interpolate(result_min, size=[255,255])
            result3 = F.interpolate(result_min_min, size=[255,255])
            # result4 = F.interpolate(result_min_min_min, size=[255, 255])

            prediction = result + result2 + result3
            # prediction = result
            # ----------------------
            # prediction = model(input)
            mse = criterion(prediction, target)
            psnr = 10 * log10(1 / mse.item())
            avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))
    psnr_y.append(avg_psnr / len(testing_data_loader))

# def checkpoint(epoch):
#     model_out_path = "model_epoch_{}.pth".format(epoch)
#     torch.save(model, model_out_path)
#     print("Checkpoint saved to {}".format(model_out_path))




for epoch in range(1, opt.nEpochs + 1):
    train(epoch)
    test()
    # train_cascade(epoch)
    # test_cascade()

    psnr_x.append(epoch)
    # checkpoint(epoch)

title_loss = "cascade_loss_" + str(opt.nEpochs) + "epoch"
title_psnr = "cascade_PSNR_" + str(opt.nEpochs) + "epoch"

# 把loss数据绘制成折线图
# 设置图片的尺寸和外框颜色 w白色
plt.figure(figsize=(10, 8), dpi=80, facecolor='w')
# 绘制图片
plt.plot(loss_x, loss_y)
# 为x y 轴和图形添加标题信息
plt.title(title_loss)
plt.xlabel("epoch")
plt.ylabel("loss_number")
plt.grid(True) ##增加格点
# 最后一个点设置数字标签
plt.text(loss_x[len(loss_x)-1], loss_y[len(loss_y)-1], loss_y[len(loss_y)-1], ha='center', va='bottom', fontsize=10)
plt.show()

# 把PSNR数据绘制成折线图
# 设置图片的尺寸和外框颜色 w白色
plt.figure(figsize=(10, 8), dpi=80, facecolor='w')
# 绘制图片
plt.plot(psnr_x, psnr_y)
# 为x y 轴和图形添加标题信息
plt.title(title_psnr)
plt.xlabel("epoch")
plt.ylabel("PSNR_number(dB)")
plt.grid(True) ##增加格点
# 最后一个点设置数字标签
plt.text(psnr_x[len(psnr_x)-1], psnr_y[len(psnr_y)-1], psnr_y[len(psnr_y)-1], ha='center', va='bottom', fontsize=10)
plt.show()

endtime = datetime.datetime.now()
print(opt.nEpochs, "epoch总共用时：")
print(endtime - starttime)
