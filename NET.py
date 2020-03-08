import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import math
from PIL import Image
import cv2
import numpy as np


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.imgL = imgL
        # self.imgR = imgR
        self.conv0 = nn.Conv2d(3, 32, 5, 1, 2)
        self.bn0 = nn.BatchNorm2d(32)
        self.res_block = self.res_layers(BasicBlock, 32, 32, 8, stride=1)
        self.conv1 = nn.Conv2d(32, 32, 3, 1, 1)

    def res_layers(self, block, in_planes, planes, num_block, stride):
        strides = [stride] + [1] * (num_block - 1)
        layers = []
        for step in strides:
            layers.append(block(in_planes, planes, step))
        return nn.Sequential(*layers)

    def forward(self, image_a, image_b, patch_size=8, iterations=5, dtresh=0.01,initialization=None, itresh=None, device=None):
        # Image manipulation
        p = int(patch_size / 2)
        if p == 0:
            p = 1
        patch_size = p * 2

        img_to_tens = transforms.ToTensor()
        padder = transforms.Pad((p, p, p - 1, p - 1), fill=0,
                                padding_mode='constant')
        image_a = image_a.convert('RGB')
        image_b = image_b.convert('RGB')
        a = img_to_tens(padder(image_a)).detach().permute(1, 2, 0).to(device)
        b = img_to_tens(padder(image_b)).detach().permute(1, 2, 0).to(device)
        shape = a.shape
        imgL = a.view(1, 3, shape[0], shape[1])
        imgR = b.view(1, 3, shape[0], shape[1])

        imgL = F.relu(self.bn0(self.conv0(imgL)))
        imgR = F.relu(self.bn0(self.conv0(imgR)))

        imgL_block = self.res_block(imgL)
        imgR_block = self.res_block(imgR)

        imgL = self.conv1(imgL_block)
        imgR = self.conv1(imgR_block)
        offsets, mapping = self. patch_match(imgL, imgR, image_a, image_b, patch_size, iterations, dtresh,initialization, itresh, device)


        return offsets, mapping

    def patch_match(self, imgL, imgR, image_a, image_b, patch_size=8, iterations=5,
                    dtresh=0.01,initialization=None, itresh=None, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        p = int(patch_size / 2)
        if p == 0:
            p = 1
        patch_size = p * 2

        # Image manipulation
        img_to_tens = transforms.ToTensor()
        padder = transforms.Pad((p, p, p - 1, p - 1), fill=0,
                                padding_mode='constant')
        image_a = image_a.convert('RGB')
        image_b = image_b.convert('RGB')
        a_w, a_h = image_a.size[0], image_a.size[1]
        b_w, b_h = image_b.size[0], image_b.size[1]

        rs_bound = min(b_h, b_w)
        rs_alpha = 1 / 2
        rs_end = int(-(math.log(rs_bound)) / math.log(rs_alpha))
        jump = patch_size

        # Patch Matrix
        a = img_to_tens(padder(image_a)).detach().permute(1, 2, 0).to(device)
        b = img_to_tens(padder(image_b)).detach().permute(1, 2, 0).to(device)

        shape = a.shape
        # a = a.view(1, 3, shape[0], shape[1])
        # b = b.view(1, 3, shape[0], shape[1])
        a = imgL
        b = imgR
        a = a.view(shape[0], shape[1], 32)
        b = b.view(shape[0], shape[1], 32)
        depth = a.shape[2]
        patch_a = a.unfold(0, patch_size, 1).unfold(1, patch_size, 1) \
            .reshape(-1, depth * ((patch_size) ** 2))

        pnum_a = patch_a.shape[0]

        # PCA
        random_indexes = torch.randint(low=0, high=pnum_a, size=(1000,)).to(device)
        random_patch_a = torch.index_select(patch_a, 0, random_indexes)
        svd = torch.svd(random_patch_a)
        sigma = svd[1]
        v = svd[2]
        svd = None
        svd_energy = (sigma ** 2).sum()
        svd_sum = sigma[0] ** 2
        limit = 1
        while ((svd_sum / svd_energy) < 0.995):
            svd_sum += sigma[limit] ** 2
            limit += 1
        patch_a = torch.matmul(patch_a, v[:, :limit])

        patch_b = b.unfold(0, patch_size, 1).unfold(1, patch_size, 1) \
            .reshape(-1, depth * ((patch_size) ** 2))
        pnum_b = patch_b.shape[0]
        patch_b = torch.matmul(patch_b, v[:, :limit])
        norm_a = torch.norm(patch_a, dim=1)
        norm_b = torch.norm(patch_b, dim=1)

        # Random initialization
        a_index = torch.arange(pnum_a, dtype=torch.int64).to(device)
        map_v = torch.randint(
            high=pnum_b, size=(pnum_a,), dtype=torch.int64).to(device)
        patch_comp = torch.index_select(patch_b, 0, map_v)
        dot_prod = torch.matmul(
            patch_a.unsqueeze(1), patch_comp.unsqueeze(2)).view(-1)
        norm_b_comp = torch.index_select(norm_b, 0, map_v)
        dist_v = norm_a ** 2 + norm_b_comp ** 2 - 2 * dot_prod
        o_v = map_v - a_index

        for i in range(1, iterations + 1):

            # Multi-scale Initialization
            if initialization is not None and itresh is not None and i == itresh:
                initialization = initialization.to(device)
                init_v = (initialization.reshape(-1, 2)[:, 0] * a_w
                          + initialization.reshape(-1, 2)[:, 1])
                map_init = a_index + init_v
                map_init = map_init.ge(0).long() * map_init
                map_init = map_init.lt(pnum_b).long() * map_init
                patch_comp = torch.index_select(patch_b, 0, map_init)
                dot_prod = torch.matmul(
                    patch_a.unsqueeze(1),
                    patch_comp.unsqueeze(2)
                ).reshape(-1)
                norm_b_comp = torch.index_select(norm_b, 0, map_init)
                dist_init = norm_a ** 2 + norm_b_comp ** 2 - 2 * dot_prod
                dist_comp = torch.cat(
                    (dist_v.unsqueeze(1), dist_init.unsqueeze(1)), dim=1)
                o_comp = torch.cat((o_v.unsqueeze(1), init_v.unsqueeze(1)), dim=1)
                dist_min = dist_comp.min(dim=1)
                o_v = torch.gather(
                    o_comp, 1, dist_min.indices.unsqueeze(1)).reshape(-1)
                dist_v = dist_min.values
                map_v = o_v + a_index

            # Scelta delle patch
            ptc = torch.masked_select(a_index, dist_v.gt(dtresh))
            ptc_num = ptc.shape[0]
            if ptc_num == 0:
                break

            # Propagation
            left = ptc.clone()
            left -= left.ge(1).long()
            right = ptc.clone()
            right += right.lt(pnum_a - 1).long()
            up = ptc.clone()
            up -= up.ge(a_w).long() * a_w
            down = ptc.clone()
            down += down.lt(pnum_a - a_w).long() * a_w

            # Jump Flooding
            jump_up = ptc.clone()
            jump_up -= jump_up.ge(a_w * jump).long() * a_w * jump
            jump_down = ptc.clone()
            jump_down += jump_down.lt(pnum_a - a_w * jump).long() * a_w * jump
            jump_left = ptc.clone()
            jump_left -= jump_left.ge(jump).long() * jump
            jump_right = ptc.clone()
            jump_right += jump_right.lt(pnum_a - jump).long() * jump

            # Matrice delle patch candidate
            a_index_comp = torch.cat((left.unsqueeze(1),
                                      right.unsqueeze(1),
                                      up.unsqueeze(1),
                                      down.unsqueeze(1),
                                      jump_up.unsqueeze(1),
                                      jump_down.unsqueeze(1),
                                      jump_left.unsqueeze(1),
                                      jump_right.unsqueeze(1)), dim=1)

            offset_comp = (torch.gather(o_v, 0, a_index_comp.contiguous().view(-1))
                           .reshape(-1, a_index_comp.shape[1]))

            # Random Search
            rs_base_off = torch.index_select(o_v, 0, ptc).to(device)
            rs_bound_v = ((rs_alpha ** torch.arange(start=1, end=rs_end).float())
                          * rs_bound)
            rs_rand_x = (torch.empty(ptc_num, rs_bound_v.shape[0]).uniform_(-1, 1)
                         * rs_bound_v.unsqueeze(0).float()).long().to(device)
            rs_rand_y = (torch.empty(ptc_num, rs_bound_v.shape[0]).uniform_(-1, 1)
                         * rs_bound_v.unsqueeze(0).float()).long().to(device)
            rs_choice = (((rs_base_off / a_w).long().unsqueeze(1) + rs_rand_x)
                         * a_w + (rs_base_off % a_w).unsqueeze(1) + rs_rand_y)

            offset_comp = torch.cat((offset_comp, rs_choice), dim=1)

            b_index_comp = ptc.unsqueeze(1) + offset_comp
            b_index_comp = b_index_comp * torch.ge(b_index_comp, 0).long()
            b_index_comp = b_index_comp * torch.lt(b_index_comp, pnum_b).long()

            pnum_comp = b_index_comp.shape[1]
            # Patch to compare matrix
            patch_b_comp = torch.index_select(patch_b, 0,
                                              b_index_comp.contiguous().view(-1))
            patch_b_comp = patch_b_comp.unfold(0, pnum_comp,
                                               pnum_comp).permute(0, 2, 1)
            patch_a_comp = torch.index_select(patch_a, 0, ptc)

            # Dot product
            dot_prod = (torch.matmul(patch_a_comp.unsqueeze(1),
                                     patch_b_comp.transpose(1, 2))
                        .view(-1, pnum_comp))
            # Selecting norms
            norm_b_comp = (torch.index_select(norm_b, 0, b_index_comp.contiguous()
                                              .view(-1)).reshape(-1, pnum_comp))
            norm_a_comp = torch.index_select(norm_a, 0, ptc)
            # Distance
            dist_comp = (norm_a_comp.unsqueeze(1)) ** 2 + norm_b_comp ** 2 - 2 * dot_prod

            dist_ptc = torch.index_select(dist_v, 0, ptc)
            map_ptc = torch.index_select(map_v, 0, ptc)
            dist_comp = torch.cat((dist_comp, dist_ptc.unsqueeze(1)), dim=1)
            b_index_comp = torch.cat((b_index_comp, map_ptc.unsqueeze(1)), dim=1)
            dist_min = dist_comp.min(dim=1)

            (map_v.scatter_(0, ptc,
                            b_index_comp.gather(1, dist_min.indices.unsqueeze(1))
                            .reshape(-1)))
            dist_v.scatter_(0, ptc, dist_min.values)
            o_v = map_v - a_index

        mapping = (torch.cat((map_v.unsqueeze(1) / a_w,
                              map_v.unsqueeze(1) % a_w), dim=1)
                   .reshape(a_h, a_w, 2).to(torch.device('cpu')))
        offsets = (mapping
                   - (torch.cat((a_index.unsqueeze(1) / a_w,
                                 a_index.unsqueeze(1) % a_w), dim=1)
                      .reshape(a_h, a_w, 2).to(torch.device('cpu'))))
        # print(offsets.shape)
        # print(mapping.shape)
        return offsets, mapping