import torch
from torchvision import transforms
import math
from PIL import Image
import cv2
import numpy as np

import NET


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
    # img = Image.open('./cup_a.jpg')
    # ref = Image.open('./cup_b.jpg')
    img = Image.open('./left.jpg')
    ref = Image.open('./right.jpg')
    #img = Image.open('./left.png')
    #ref = Image.open('./right.png')
    # width = img.size[0]
    # height = img.size[1]
    # print(width)
    net = NET.Net()
    devices = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # devices = torch.cuda.current_device()
    net = net.to(devices)
    offsets, mapping = multi_scale_pm(net,img, ref)
    print(mapping)

    # scale_factor = 0.5
    # a_h_s, a_w_s = int(img.size[1] * scale_factor), int(img.size[0] * scale_factor)
    # b_h_s, b_w_s = int(ref.size[1] * scale_factor), int(ref.size[0] * scale_factor)
    # img_scaled = transforms.functional.resize(img, (a_h_s, a_w_s))
    # ref_scaled = transforms.functional.resize(ref, (b_h_s, b_w_s))
    # print(type(img_scaled))
    # print(type(ref_scaled))