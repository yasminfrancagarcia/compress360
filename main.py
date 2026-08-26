from PIL import Image
import matplotlib.pyplot as plt
import json

caminhoJson = "benchmark_teste500Sun.json" 
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



def fazer_grafico2():
    with open(caminhoJson) as f:
        data = json.load(f)

    bpps = data["results"]["bpp"]
    psnrs = data["results"]["psnr"]
    ms_ssim_db = data["results"]["ms-ssim-db"]
    ws_psnr = data["results"]["ws-psnr"]

    # Pegar somente o 1º, 3º e último checkpoint
    indices = [0, 2, 5]

    bpps = [bpps[i] for i in indices]
    psnrs = [psnrs[i] for i in indices]
    ms_ssim_db = [ms_ssim_db[i] for i in indices]
    ws_psnr = [ws_psnr[i] for i in indices]

    # gráfico bpp x PSNR
    plt.plot(bpps, psnrs, marker="o", label="LALIC")
    plt.xlabel("Bitrate (bpp)")
    plt.ylabel("PSNR (dB)")
    plt.title("Curva Rate-Distortion — bpp x PSNR")
    plt.legend()
    plt.grid(True)
    plt.show()

    # gráfico bpp x MS-SSIM
    plt.plot(bpps, ms_ssim_db, marker="o", label="LALIC")
    plt.xlabel("Bitrate (bpp)")
    plt.ylabel("MS-SSIM (dB)")
    plt.title("Curva Rate-Distortion — bpp x MS-SSIM")
    plt.legend()
    plt.grid(True)
    plt.show()

    # gráfico bpp x W-PSNR
    plt.plot(bpps, ws_psnr, marker="o", label="LALIC")
    plt.xlabel("Bitrate (bpp)")
    plt.ylabel("W-PSNR (dB)")
    plt.title("Curva Rate-Distortion — bpp x W-PSNR")
    plt.legend()
    plt.grid(True)
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

caminhos = [
    "benchmark_test500_0025_20epoctrain30.json",
    "benchmark_test500_0067_20epoc_train30.json",
    "benchmark_test500_0483_20epoc_train30.json"
]


def fazer_grafico3(caminhosJson):
    bpps = []
    psnrs = []
    ms_ssim_db = []
    ws_psnr = []

    # Ler os 3 JSONs
    for caminho in caminhosJson:
        with open(caminho) as f:
            data = json.load(f)

        bpps.append(data["results"]["bpp"][0])
        psnrs.append(data["results"]["psnr"][0])
        ms_ssim_db.append(data["results"]["ms-ssim-db"][0])
        ws_psnr.append(data["results"]["ws-psnr"][0])

    #ordenar pelo bpp para a curva ficar correta
    dados = sorted(zip(bpps, psnrs, ms_ssim_db, ws_psnr))

    bpps, psnrs, ms_ssim_db, ws_psnr = zip(*dados)

    #gráfico bpp x PSNR
    plt.figure()
    plt.plot(bpps, psnrs, marker="o", label="LALIC")
    plt.xlabel("Bitrate (bpp)")
    plt.ylabel("PSNR (dB)")
    plt.title("Curva Rate-Distortion — bpp x PSNR")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Gráfico bpp x MS-SSIM
    plt.figure()
    plt.plot(bpps, ms_ssim_db, marker="o", label="LALIC")
    plt.xlabel("Bitrate (bpp)")
    plt.ylabel("MS-SSIM (dB)")
    plt.title("Curva Rate-Distortion — bpp x MS-SSIM")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Gráfico bpp x W-PSNR
    plt.figure()
    plt.plot(bpps, ws_psnr, marker="o", label="LALIC")
    plt.xlabel("Bitrate (bpp)")
    plt.ylabel("W-PSNR (dB)")
    plt.title("Curva Rate-Distortion — bpp x W-PSNR")
    plt.legend()
    plt.grid(True)
    plt.show()

#comparar_todas_qualidades("517.jpg")

#fazer_grafico()

#fazer_grafico3(caminhos)

import fiftyone.utils.openimages as fou

# Obtém a lista completa de classes do Open Images v7
all_classes = fou.get_classes()

# Filtra por 'bed' ou 'room'
classes_desejadas = [
    "Tree", "Bench", "Fountain",
    "House", "Building", "Porch",
    "Bed", "Couch", "Window", "Kitchen appliance",
    "Beaker", "Medical equipment", "Computer monitor",
    "Car", "Airplane", "Bus"
]
keywords =  [
    "Tree", "Bench", "Fountain",
        "House", "Building", "Porch",
        "Bed", "Couch", "Window", "Kitchen appliance",
        "Beaker", "Medical equipment", "Computer monitor",
        "Car", "Airplane", "Bus"
]
matching_classes = [
    cls for cls in all_classes 
    if any(kw in cls for kw in keywords)
]

# Exibe o resultado
print(f"Total de classes encontradas: {len(matching_classes)}\n")
print("Classes disponíveis:")
for cls in sorted(matching_classes):
    print(f"- {cls}")

# print(f"Total de classes encontradas: {len(all_classes)}\n")
# for cls in sorted(all_classes):
#     print(f"- {cls}")