import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# usar times new roman
plt.rcParams["font.family"] = "Times New Roman"

# aumentar o tamanho das fontes
plt.rcParams["font.size"] = 20
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["axes.labelsize"] = 20
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
plt.rcParams["legend.fontsize"] = 16


# ------------------------------------------------------------
# lalic original
# ------------------------------------------------------------
lalic_pesos_pre_treinados = {
    "bpp": np.array([0.128777, 0.328257, 0.938292]),
    "ws_psnr": np.array([27.972017, 31.224781, 36.833692]),
    "ws_ssim": np.array([0.786747, 0.878492, 0.960044])
}

# lalic baseline 30k planas
lalic_baseline_30k = {
    "bpp": np.array([0.158131, 0.326836, 1.025072]),
    "ws_psnr": np.array([28.139117, 30.424208, 35.586628]),
    "ws_ssim": np.array([0.804663, 0.87114, 0.954659])
}

# estratégia I
estrategia_1 = {
    "bpp": np.array([0.157097, 0.326110, 1.038092]),
    "ws_psnr": np.array([28.269910, 30.664554, 36.281543]),
    "ws_ssim": np.array([0.806381, 0.873914, 0.958194])
}

# estratégia II
estrategia_2 = {
    "bpp": np.array([0.166709, 0.338719, 1.065430]),
    "ws_psnr": np.array([28.581836, 30.904750, 36.497981]),
    "ws_ssim": np.array([0.816686, 0.881484, 0.961196])
}

# estratégia III
estrategia_3 = {
    "bpp": np.array([0.168745, 0.335549, 1.029863]),
    "ws_psnr": np.array([28.778219, 31.197338, 37.000000]),
    "ws_ssim": np.array([0.820905, 0.884617, 0.963558])
}

# estratégia IV
estrategia_4 = {
    "bpp": np.array([0.160812, 0.323918, 1.021573]),
    "ws_psnr": np.array([28.577817, 30.876526, 35.40396]),
    "ws_ssim": np.array([0.815582, 0.879992, 0.955262])
}

# marker diferente por método, todas com linha solida (sem pontilhado/tracejado)
metodos = [
    ("LALIC",  lalic_baseline_30k,        "s"),
    ("Estratégia I",    estrategia_1,              "^"),
    ("Estratégia II",   estrategia_2,              "v"),
    ("Estratégia III",  estrategia_3,              "D"),
    ("Estratégia IV",   estrategia_4,              "P"),
]

fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))


def plot_metric(ax, chave_y, ylabel, titulo):
    for nome, dados, marker in metodos:
        ordem = np.argsort(dados["bpp"])
        bpp = dados["bpp"][ordem]
        y = dados[chave_y][ordem]

        ax.plot(
            bpp,
            y,
            marker=marker,
            markersize=3,
            linewidth=0.8,
            markeredgewidth=0.8,
            linestyle="-",
            label=nome
        )

    ax.set_xlabel("bpp")
    ax.set_ylabel(ylabel)
    ax.set_title(titulo)

    ax.xaxis.set_major_locator(MultipleLocator(0.1))

    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", frameon=True, ncol=1, fontsize=16)


plot_metric(axes[0], "ws_psnr", "WS-PSNR (dB)", "")
plot_metric(axes[1], "ws_ssim", "WS-SSIM", "")

plt.tight_layout()

plt.savefig("rd_curves_ws_final.png", dpi=300, bbox_inches="tight")
plt.savefig("rd_curves_ws_final.pdf", bbox_inches="tight")

plt.close()