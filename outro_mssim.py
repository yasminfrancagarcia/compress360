import torch
import torch.nn.functional as F

def _gaussian_kernel(win_size=11, sigma=1.5, device=None, dtype=None):
    coords = torch.arange(win_size, dtype=dtype, device=device) - win_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return torch.outer(g, g)

def ssim_map(a, b, win_size=11, sigma=1.5, data_range=1.0, K=(0.01, 0.03)):
    C = a.size(1)
    kernel = _gaussian_kernel(win_size, sigma, a.device, a.dtype).expand(C, 1, win_size, win_size)
    pad = win_size // 2

    mu_a = F.conv2d(a, kernel, padding=pad, groups=C)
    mu_b = F.conv2d(b, kernel, padding=pad, groups=C)
    mu_a_sq, mu_b_sq, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b

    sigma_a_sq = F.conv2d(a * a, kernel, padding=pad, groups=C) - mu_a_sq
    sigma_b_sq = F.conv2d(b * b, kernel, padding=pad, groups=C) - mu_b_sq
    sigma_ab   = F.conv2d(a * b, kernel, padding=pad, groups=C) - mu_ab

    C1, C2 = (K[0] * data_range) ** 2, (K[1] * data_range) ** 2
    ssim_n = (2 * mu_ab + C1) * (2 * sigma_ab + C2)
    ssim_d = (mu_a_sq + mu_b_sq + C1) * (sigma_a_sq + sigma_b_sq + C2)
    return ssim_n / ssim_d  # (1, C, H, W) — mesma resolução da entrada, graças ao padding='same'


def compute_ws_ssim(a, b):
    smap = ssim_map(a, b)
    weights = get_latitude_weights(smap.size(2), device=a.device, dtype=a.dtype)  # reaproveita a mesma função
    wssim = (smap * weights).sum() / (weights.sum() * smap.size(1) * smap.size(3))
    return wssim.item()