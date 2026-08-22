from PIL import Image
import matplotlib.pyplot as plt
import json

caminhoJson = "benchmark_test500_train30.json" 
checkpoints = [
    "checkpoints/lalic-q1.pth",
    "checkpoints/lalic-q2.pth",
    "checkpoints/lalic-q3.pth",
    "checkpoints/lalic-q4.pth",
    "checkpoints/lalic-q5.pth",
    "checkpoints/lalic-q6.pth"
]
qualidades = ["1", "2", "3", "4", "5", "6"]

def comparar_todas_qualidades(nome, qualidades=range(1, len(checkpoints)+1)):
    n = len(list(qualidades)) + 1
    fig, axs = plt.subplots(1, n, figsize=(4*n, 5))

    orig = Image.open(f"sun360test_500/{nome}")
    axs[0].imshow(orig); axs[0].set_title(f"Original\n{nome}"); axs[0].axis("off")

    for i, q in enumerate(qualidades, start=1):
        recon = Image.open(f"recon_imagesSun360_test/{q}/{nome}")
        axs[i].imshow(recon); axs[i].set_title(f"q={q}"); axs[i].axis("off")

    plt.tight_layout()
    plt.show()


def fazer_grafico():
    with open(caminhoJson) as f:
        data = json.load(f)

    bpps = data["results"]["bpp"]
    psnrs = data["results"]["psnr"]
    ms_ssim_db = data["results"]["ms-ssim-db"]
    ws_psnr = data["results"]["ws-psnr"]

    # grafico de bpp x psnr
    plt.plot(bpps, psnrs, marker="o", label="LALIC")
    plt.xlabel("Bitrate (bpp)")
    plt.ylabel("PSNR (dB)")
    plt.title("Curva Rate-Distortion — bpp x psnr")
    plt.legend()
    plt.grid(True)
    plt.show()


    # grafico de bpp x ms-ssim-db   
    plt.plot(bpps, ms_ssim_db, marker="o", label="LALIC")
    plt.xlabel("Bitrate (bpp)")
    plt.ylabel("ms-ssim (dB)")
    plt.title("Curva Rate-Distortion — bpp x ms-ssim-db")
    plt.legend()
    plt.grid(True)
    plt.show()

    # grafico de bpp x ws-psnr
    plt.plot(bpps, ws_psnr, marker="o", label="LALIC")
    plt.xlabel("Bitrate (bpp)")
    plt.ylabel("W-PSNR (dB)")
    plt.title("Curva Rate-Distortion — bpp x w-psnr")
    plt.legend()
    plt.grid(True)
    plt.show()




#comparar_todas_qualidades("517.jpg")

fazer_grafico()