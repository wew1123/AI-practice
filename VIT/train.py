import torch
from torch import nn, einsum
import torch.nn.functional as F
from torchvision.transforms import transforms
import torch.optim as optim

import pandas as pd
from pandas.core.frame import DataFrame
from matplotlib import pyplot as plt
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import DataLoader

from vit import *

# 这个是训练对日期识别的
# 读入日期标签
# 读入裁切好的日期图片
pos_path = "/Users/vulpix/Downloads/jiutian1-master/交付测试报告-训练/crop/"
img_path = "/Users/vulpix/Downloads/jiutian1-master/交付测试报告-训练/语音专线/"

def nums(num):
    if num<10:
        nums = "000" + "00" + str(num)
    elif (num < 100):
        nums = "000" + "0" + str(num)
    else:
        nums = "000" + str(num)

    return nums


def getImg_s(img_path=img_path, start=1, end=59):
    img = Image.open(img_path +nums(start) + ".jpg")
    img = transform(img)
    img = img.view(-1, 3, 128, 128)
    img_s = img
    for num in range(start + 1, end + 1):
        try:
            img_p = img_path + nums(num) + ".jpg"
            img = Image.open(img_p)
            img = transform(img)
            img = img.view(-1, 3, 128, 128)
            img_s = torch.cat((img_s, img), dim=0)
        except:
            continue
    return img_s


def backward(data_in, label_in,):
    optimizer.zero_grad()
    print(label_in.device)
    print(data_in.device)
    print(model.device())

    out = model(data_in)
    loss = criterion(out, label_in)
    loss.backward()
    optimizer.step()
    return loss


def train(dataloader,epoch=10):
    min = 0
    for i in range(epoch):
        loss = 0
        i+=1
        for data_in, label_in in dataloader:
            label_in = label_in.float().to("mps")
            data_in = data_in.float().to("mps")
            loss += backward(data_in=data_in, label_in=label_in)
            if i % 5 == 0:
                print(loss)
        if i % 50 == 0:
            print("latest:  ", loss)
            torch.save(model, "pt/latest")
        if loss < min:
            min = loss
            torch.save(model, "pt/latest")


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize([128, 128]),
        transforms.ToTensor(),
        # transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 常用标准化
    ])
    model = torch.load("pt/best")
    model = model.to("mps")

    label = torch.load(pos_path + "pos_label")
    img_s = getImg_s()
    train_data = list(zip(img_s, label))


    batch_size = 32
    data_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=3, drop_last=False)
    lr = 6e-5
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.8)  # 1e-4
    criterion = nn.MultiLabelSoftMarginLoss()
    epo = 0
    while(True):

        train(data_loader,epoch=1000)

        epo += 1
        lr = lr * 0.95
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.8)
        if(epo==65):
            lr = 6e-5
        if(epo == 120):
            epo = 30