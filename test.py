from pathlib import Path
import argparse
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import cv2
from dataset import StereoDataset
from dpn import DPN
from utils import *


def parse_args():
    parser = argparse.ArgumentParser(description='MaterialStereo')
    parser.add_argument('--data-path', type=str, default='data')
    parser.add_argument('--list-path', type=str, default='lists')
    parser.add_argument('--ckpt-path', type=str, default='ckpt')
    parser.add_argument('--result-path', type=str, default='result')
    parser.add_argument('--test-split', type=str, default='20170222_0951,20170222_1423,20170223_1639,20170224_0742')
    parser.add_argument('--black-level', type=float, default=2.0)
    parser.add_argument('--clamp-value', type=float, default=5.0)
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--vis-maxd', type=int, default=0.031)
    opt = parser.parse_args()
    opt.test_split = opt.test_split.split(',')
    return opt


def test(opt, test_loader, dpnet):
    dpnet = dpnet.eval()
    ans = [[],[],[],[],[],[],[],[]]
    for iteration, batch in enumerate(test_loader):
        collection, key, raw_rgb, raw_nir, _, _, _, _ = batch
        rgb = F.relu((raw_rgb - opt.black_level) / (255.0 - opt.black_level)).cuda()
        nir = F.relu((raw_nir - opt.black_level) / (255.0 - opt.black_level)).cuda()
        rgb_ratio = 0.5 / (rgb.mean(1).mean(1).mean(1) + 1e-3)
        nir_ratio = 0.5 / (nir.mean(1).mean(1).mean(1) + 1e-3)
        rgb = torch.clamp(rgb * rgb_ratio.view(-1, 1, 1, 1), 0.0, opt.clamp_value)
        nir = torch.clamp(nir * nir_ratio.view(-1, 1, 1, 1), 0.0, opt.clamp_value)
        ldisps, rdisps = dpnet(rgb, nir)
        for i in range(rgb.shape[0]):
            invd = cpu_np(ldisps[0][i,0])
            if opt.vis:
                show_ldisp = to_image(ldisps[0][i] / opt.vis_maxd)
                show_image = show_ldisp.transpose([1, 2, 0])
                show_image = cv2.applyColorMap(show_image, cv2.COLORMAP_JET)
                cv2.imwrite(str(Path(opt.result_path) / 'pngs' / (key[i] + '.png')), show_image)

            f = open(Path(opt.data_path) / collection[i] / 'Keypoint' / (key[i] + '_Keypoint.txt'), 'r')
            gts = f.readlines()
            f.close()
            for gt in gts:
                x, y, d, c = gt.split()
                x = round(float(x) * 582) - 1
                x = int(max(0,min(582, x)))
                y = round(float(y) * 429) - 1
                y = int(max(0,min(429, y)))
                d = float(d) * 582
                c = int(c)
                p = max(0, invd[y, x] * 582)
                ans[c].append((p-d)*(p-d))
    rmse = []
    for c in range(8):
        rmse.append(pow(sum(ans[c]) / len(ans[c]), 0.5))
    print('Common    Light     Glass     Glossy  Vegetation   Skin    Clothing    Bag       Mean')
    print(round(rmse[0], 4), '  ', round(rmse[1], 4), '  ', round(rmse[2], 4), '  ', round(rmse[3], 4), '  ', round(rmse[4], 4), '  ', round(rmse[5], 4), '  ', round(rmse[6], 4), '  ', round(rmse[7], 4), '  ', round(sum(rmse) / 8.0, 4))
    print()
    if opt.vis:
        f = open(Path(opt.result_path) / 'performance.txt', 'w')
        print('Common    Light     Glass     Glossy  Vegetation   Skin    Clothing    Bag       Mean', file=f)
        print(round(rmse[0], 4), '  ', round(rmse[1], 4), '  ', round(rmse[2], 4), '  ', round(rmse[3], 4), '  ', round(rmse[4], 4), '  ', round(rmse[5], 4), '  ', round(rmse[6], 4), '  ', round(rmse[7], 4), '  ', round(sum(rmse) / 8.0, 4), file=f)
        f.close()


if __name__ == '__main__':
    cv2.setNumThreads(0)
    opt = parse_args()
    print(opt)

    if opt.vis:
        (Path(opt.result_path) / 'pngs').mkdir(parents=True, exist_ok=True)

    test_set = StereoDataset(opt.data_path, opt.list_path, opt.test_split)
    test_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=False)

    dpnet = DPN(in_shape=(test_set.height, test_set.width))

    checkpoint = torch.load(opt.ckpt_path)
    dpnet.load_state_dict(checkpoint['dpnet'])

    dpnet = dpnet.cuda()
    dpnet = nn.DataParallel(dpnet)

    test(opt, test_loader, dpnet)
