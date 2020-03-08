import torch
from torchvision import transforms
import math
from PIL import Image
import cv2
import numpy as np
import torch.optim as optim
import torch.nn as nn

import NET
from python_pfm import readPFM


def multi_scale_pm(net,img, ref, patch_size=8, iterations=5, dtresh=0.01, itresh=1, device=None):
    print(type(img))
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if min(img.size[0], img.size[1], ref.size[0], ref.size[1]) <= 300:
        return net(img, ref, patch_size,iterations=iterations, device=device)

    scale_factor = 0.5
    a_h_s, a_w_s = int(img.size[1] * scale_factor), int(img.size[0] * scale_factor)
    b_h_s, b_w_s = int(ref.size[1] * scale_factor), int(ref.size[0] * scale_factor)
    img_scaled = transforms.functional.resize(img, (a_h_s, a_w_s))
    ref_scaled = transforms.functional.resize(ref, (b_h_s, b_w_s))
    p_size_scaled = min(int(patch_size * scale_factor), 2)
    init_scaled = multi_scale_pm(net,img_scaled, ref_scaled, p_size_scaled,
                                      iterations, dtresh, itresh,
                                      device=device)[0].permute(2, 0, 1).float()
    # print(init_scaled.shape)
    upsampler = torch.nn.Upsample(size=(img.size[1], img.size[0]))
    init = ((upsampler(init_scaled.unsqueeze(0)).squeeze(0)
             .permute(1, 2, 0) * (1 / scale_factor)).long())
    return net(img, ref, patch_size, iterations, dtresh=dtresh,
                       initialization=init, itresh=itresh, device=device)


if __name__ == '__main__':
    dispL = readPFM('./disp.pfm')[0].astype(np.uint8).reshape(540, 960, 1).transpose((2, 0, 1))
    dispL = dispL.reshape((1, 1, 540, 960))
    dispL = torch.from_numpy(dispL)
    dispL = dispL.cuda()
    # img = Image.open('./cup_a.jpg')
    # ref = Image.open('./cup_b.jpg')
    img = Image.open('./left.jpg')
    ref = Image.open('./right.jpg')
    #img = Image.open('./left.png')
    #ref = Image.open('./right.png')
    width = img.size[0]
    height = img.size[1]
    net = NET.Net()
    devices = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net.to(devices)
    loss_fn = nn.L1Loss()
    optimizer = optim.RMSprop(net.parameters(), lr=0.001, alpha=0.9)
    for i in range(50):
        net.train()
        net.zero_grad()
        optimizer.zero_grad()
        offsets, mapping = multi_scale_pm(net,img, ref)
        flag = True
        count = 0
        temp = np.ones((height, width)) * -1.5
        while (flag):
            count += 1
            for i in range(height):
                for j in range(width):
                    if temp[i, j] == -1.5:
                        if mapping[i, j][0] == i and j - mapping[i, j][1] > 0:
                            temp[i, j] = j - mapping[i, j][1]
                        else:
                            temp[i, j] = 0.0
            if count == 1:
                flag = False
        x = temp.view(1, 1, height, width)
        tt = loss_fn(x, dispL)
        tt.backward()
        optimizer.step()

    # scale_factor = 0.5
    # a_h_s, a_w_s = int(img.size[1] * scale_factor), int(img.size[0] * scale_factor)
    # b_h_s, b_w_s = int(ref.size[1] * scale_factor), int(ref.size[0] * scale_factor)
    # img_scaled = transforms.functional.resize(img, (a_h_s, a_w_s))
    # ref_scaled = transforms.functional.resize(ref, (b_h_s, b_w_s))
    # print(type(img_scaled))
    # print(type(ref_scaled))