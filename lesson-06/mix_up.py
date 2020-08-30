import numpy as np
import random
import cv2

import torch
import torch.optim as optim
from torchvision import datasets, transforms
import argparse as args

for i,(images,target) in enumerate(train_loader):
    # 1.input output
    images = images.cuda(non_blocking=True)
    target = torch.from_numpy(np.array(target)).float().cuda(non_blocking=True)

    # 2.mixup
    alpha=config.alpha
    lam = np.random.beta(alpha,alpha)
   #randperm返回1~images.size(0)的一个随机排列
    index = torch.randperm(images.size(0)).cuda()
    inputs = lam*images + (1-lam)*images[index,:]
    targets_a, targets_b = target, target[index]
    outputs = model(inputs)
    loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)

    # 3.backward
    optimizer.zero_grad()   # reset gradient
    loss.backward()
    optimizer.step()        # update parameters of net