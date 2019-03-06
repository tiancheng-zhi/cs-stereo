import torch
import torch.nn.functional as F
import numpy as np

def cpu_np(tensor):
    return tensor.cpu().detach().numpy()


def to_image(matrix):
    image = cpu_np(torch.clamp(matrix, 0, 1) * 255).astype(np.uint8)
    if matrix.size()[0] == 1:
        image = np.concatenate((image, image, image), 0)
    return image


def anaglyph(left, right):
    return to_image(torch.cat((left, right, right), 0))


def color(im):
    im_sum = im.abs().sum(1, keepdim=True)
    im = im / (im_sum + 0.001)
    return im


def detach_pyramid(vars):
    return [var.detach() for var in vars]


def pyramid(im, n_levels=4, anti_aliasing=False):
    _, _, height, width = im.size()
    ims = [im]
    for i in range(1, n_levels):
        h = height // (2 ** i)
        w = width // (2 ** i)
        if anti_aliasing:
            im = gaussian(im)
        resized = F.interpolate(im, (h, w), mode='bilinear')
        ims.append(resized)
    return ims


def warp(im, disp):
    theta = torch.Tensor(np.array([[1, 0, 0], [0, 1, 0]])).cuda()
    theta = theta.expand((disp.size()[0], 2, 3)).contiguous()
    grid = F.affine_grid(theta, disp.size())
    disp = disp.transpose(1, 2).transpose(2, 3)
    disp = torch.cat((disp, torch.zeros(disp.size()).cuda()), 3)
    grid = grid + 2 * disp
    sampled = F.grid_sample(im, grid)
    return sampled


def warp_pyramid(ims, disps, sgn):
    result = []
    for i, im in enumerate(ims):
        disp = sgn * disps[i]
        result.append(warp(im, disp))
    return result


def fliplr(im):
    return torch.flip(im, [3,])


def fliplr_pyramid(ims):
    result = [fliplr(im) for im in ims]
    return result


def l1_loss(a, b):
    return (a - b).abs()


def l1_mean(im):
    return im.abs().mean(1, keepdim=True)


def sobel(im):
    c = im.size()[1]
    fx = torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    fx = fx.view(1, 1, 3, 3).expand(1, c, 3, 3)
    fy = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    fy = fy.view(1, 1, 3, 3).expand(1, c, 3, 3)
    if im.is_cuda:
        fx = fx.cuda()
        fy = fy.cuda()
    gradx = F.pad(F.conv2d(im, fx), (1, 1, 1, 1))
    grady = F.pad(F.conv2d(im, fy), (1, 1, 1, 1))
    return gradx, grady


def gaussian(im):
    smooth = 1/16 * im[:, :, :-2,  :-2] + 1/8 * im[:, :, 1:-1,  :-2] + 1/16 * im[:, :, 2:,  :-2] + \
             1/8  * im[:, :, :-2, 1:-1] + 1/4 * im[:, :, 1:-1, 1:-1] + 1/8  * im[:, :, 2:, 1:-1] + \
             1/16 * im[:, :, :-2, 2:  ] + 1/8 * im[:, :, 1:-1, 2:  ] + 1/16 * im[:, :, 2:, 2:  ]
    smooth = F.pad(smooth, (1, 1, 1, 1), mode='replicate')
    return smooth


def grad(im):
    gradx = F.pad((im[:, :, :, 2:] - im[:, :, :, :-2]) / 2.0, (1, 1, 0, 0))
    grady = F.pad((im[:, :, 2:, :] - im[:, :, :-2, :]) / 2.0, (0, 0, 1, 1))
    return gradx, grady


def grad_noconf(im):
    im_right = im[:, :, :, 2:]
    im_left = im[:, :, :, :-2]
    im_down = im[:, :, 2:, :]
    im_up = im[:, :, :-2, :]

    gradx = F.pad((0.5 * (im_right - im_left)), (1, 1, 0, 0)) / 2.0
    grady = F.pad((0.5 * (im_down - im_up)), (0, 0, 1, 1)) / 2.0

    return gradx, grady


def grad_conf(im, conf):
    conf = F.relu(conf.detach() - 1e-10) + 1e-10
    conf_right = conf[:, :, :, 2:]
    conf_left = conf[:, :, :, :-2]
    conf_down = conf[:, :, 2:, :]
    conf_up = conf[:, :, :-2, :]

    im_right = im[:, :, :, 2:]
    im_left = im[:, :, :, :-2]
    im_down = im[:, :, 2:, :]
    im_up = im[:, :, :-2, :]

    imd = im.detach()
    imd_right = imd[:, :, :, 2:]
    imd_left = imd[:, :, :, :-2]
    imd_down = imd[:, :, 2:, :]
    imd_up = imd[:, :, :-2, :]

    gradx = F.pad((conf_right * (imd_right - im_left) + conf_left * (im_right - imd_left)) / (conf_right + conf_left), (1, 1, 0, 0)) / 2.0
    grady = F.pad((conf_down * (imd_down - im_up) + conf_up * (im_down - imd_up)) / (conf_down + conf_up), (0, 0, 1, 1)) / 2.0

    return gradx, grady


def smooth_noconf(im):
    gradx, grady = grad_noconf(im)
    smooth = gradx.abs() + grady.abs()
    return smooth


def smooth_conf(im, conf):
    gradx, grady = grad_conf(im, conf)
    smooth = gradx.abs() + grady.abs()
    return smooth


def dssim(im_a, im_b, ksize=5):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu_a = F.avg_pool2d(im_a, ksize, 1)
    mu_b = F.avg_pool2d(im_b, ksize, 1)
    sigma_a = F.avg_pool2d(im_a ** 2, ksize, 1) - mu_a ** 2
    sigma_b = F.avg_pool2d(im_b ** 2, ksize, 1) - mu_b ** 2
    sigma_ab = F.avg_pool2d(im_a * im_b, ksize, 1) - mu_a * mu_b
    SSIM_n = (2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)
    SSIM_d = (mu_a ** 2 + mu_b ** 2 + C1) * (sigma_a + sigma_b + C2)
    SSIM = SSIM_n / SSIM_d
    pad = ksize // 2
    dssim = F.pad(torch.clamp((1 - SSIM) / 2.0, 0, 1), (pad, pad, pad, pad))
    return dssim

