import os
import torch
import torch.nn.functional as F
import math
from eval import read_image, reglob_collect_images  # reaproveita o que já existe

def match_pairs(orig_dir, recon_dir):
    orig_files = reglob_collect_images(orig_dir)
    pairs = []
    for orig_path in orig_files:
        name = os.path.basename(orig_path)
        recon_path = os.path.join(recon_dir, name)
        if os.path.isfile(recon_path):
            pairs.append((orig_path, recon_path))
        else:
            print(f"Aviso: sem par reconstruído para {name}")
    return pairs

def _gaussian_window(win_size, sigma, device, dtype):
    coords = torch.arange(win_size, dtype=dtype, device=device) - win_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g

def _ssim_map(a, b, win_size=11, sigma=1.5, data_range=1.0, K=(0.01, 0.03)):
    # a, b: (N, C, H, W). Retorna ssim_map e cs_map com o mesmo H, W (padding 'same').
    device, dtype = a.device, a.dtype
    C1, C2 = (K[0] * data_range) ** 2, (K[1] * data_range) ** 2

    g1d = _gaussian_window(win_size, sigma, device, dtype)
    win = (g1d[:, None] * g1d[None, :])[None, None]  # (1,1,win,win)
    win = win.expand(a.size(1), 1, win_size, win_size)
    pad = win_size // 2

    def filt(x):
        return F.conv2d(x, win, padding=pad, groups=a.size(1))

    mu1, mu2 = filt(a), filt(b)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = filt(a * a) - mu1_sq
    sigma2_sq = filt(b * b) - mu2_sq
    sigma12 = filt(a * b) - mu1_mu2

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map
    return ssim_map, cs_map


def _latitude_weight_map(H, W, device, dtype):
    j = torch.arange(H, device=device, dtype=dtype)
    lat = (j + 0.5 - H / 2.0) * math.pi / H
    w = torch.cos(lat).clamp(min=0)
    w = w / w.mean()
    return w.view(1, 1, H, 1).expand(1, 1, H, W)

def compute_wms_ssim_db(a, b, win_size=11, sigma=1.5):
    wms_ssim = _wms_ssim_raw(a, b, win_size, sigma)
    return -10 * math.log10(1 - wms_ssim)


def _wms_ssim_raw(a, b, win_size=11, sigma=1.5):
    weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333],
                            device=a.device, dtype=a.dtype)
    levels = weights.numel()
    mcs = []
    x, y = a, b
    for i in range(levels):
        ssim_map, cs_map = _ssim_map(x, y, win_size, sigma)
        H, W = ssim_map.shape[-2:]
        w = _latitude_weight_map(H, W, a.device, a.dtype)
        if i < levels - 1:
            mcs.append((cs_map * w).mean().clamp(min=1e-6))
            x = F.avg_pool2d(x, kernel_size=2)
            y = F.avg_pool2d(y, kernel_size=2)
        else:
            final_ssim = (ssim_map * w).mean().clamp(min=1e-6)

    mcs_stack = torch.stack(mcs + [final_ssim])
    return torch.prod(mcs_stack ** weights).item()

def evaluate_folder(orig_dir, recon_dir, device="cuda"):
    pairs = match_pairs(orig_dir, recon_dir)

    if len(pairs) == 0:
        print(f"Nenhum par encontrado em {recon_dir}")
        return None

    total = 0.0
    valid = 0

    with torch.no_grad():
        for orig_path, recon_path in pairs:

            x = read_image(orig_path).unsqueeze(0).to(device)
            x_hat = read_image(recon_path).unsqueeze(0).to(device)

            if x.shape != x_hat.shape:
                print(
                    f"Dimensões diferentes: "
                    f"{os.path.basename(orig_path)}"
                )
                continue

            score = compute_wms_ssim_db(x, x_hat)

            total += score
            valid += 1

    if valid == 0:
        return None

    return total / valid

def main(orig_dir, recon_dirs, device="cuda"):

    results = {}

    for label, recon_dir in recon_dirs.items():

        print(f"\nCalculando: {label}")

        mean_score = evaluate_folder(
            orig_dir,
            recon_dir,
            device
        )

        if mean_score is not None:
            results[label] = mean_score

            print(
                f"{label}: "
                f"{mean_score:.4f} dB"
            )

    print("\n===== RESULTADOS =====")

    for label, score in results.items():
        print(f"{label}: {score:.4f} dB")



if __name__ == "__main__":

    orig_dir = "sun360test_500"

    recon_dirs = {
        "w-mse + crop witdh 3": "recon_test500_0067_wmse_cropwidth_20epoc/3"    
    }

    main(
        orig_dir,
        recon_dirs,
        "cuda"
    )