import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ============================================================
# configuração
# ============================================================

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 14
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["xtick.labelsize"] = 13
plt.rcParams["ytick.labelsize"] = 13
plt.rcParams["legend.fontsize"] = 11

# ============================================================
# dados
# ============================================================

# ------------------------------------------------------------
# lalic original
# ------------------------------------------------------------
lalic_pesos_pre_treinados = {
    "bpp": np.array([
        0.128777,
        0.328257,
        0.938292
    ]),
    "ws_psnr": np.array([
        27.972017,
        31.224781,
        36.833692
    ]),
    "ws_ssim": np.array([
        0.786747,
        0.878492,
        0.960044
    ])
}

# ------------------------------------------------------------
# lalic baseline 30k planas
# ------------------------------------------------------------
lalic_baseline_30k = {
    "bpp": np.array([
        0.158131,
        0.326836,
        1.025072
    ]),
    "ws_psnr": np.array([
        28.139117,
        30.424208,
        35.586628
    ]),
    "ws_ssim": np.array([
        0.804663,
        0.871140,
        0.954659
    ])
}

# ------------------------------------------------------------
# estratégia I
# ------------------------------------------------------------
estrategia_1 = {
    "bpp": np.array([
        0.157097,
        0.326110,
        1.038092
    ]),
    "ws_psnr": np.array([
        28.269910,
        30.664554,
        36.281543
    ]),
    "ws_ssim": np.array([
        0.806381,
        0.873914,
        0.958194
    ])
}

# ------------------------------------------------------------
# estratégia II
# ------------------------------------------------------------
estrategia_2 = {
    "bpp": np.array([
        0.166709,
        0.338719,
        1.065430
    ]),
    "ws_psnr": np.array([
        28.581836,
        30.904750,
        36.497981
    ]),
    "ws_ssim": np.array([
        0.816686,
        0.881484,
        0.961196
    ])
}

# ------------------------------------------------------------
# estratégia III
# ------------------------------------------------------------
estrategia_3 = {
    "bpp": np.array([
        0.168745,
        0.335549,
        1.029863
    ]),
    "ws_psnr": np.array([
        28.778219,
        31.197338,
        37.000000
    ]),
    "ws_ssim": np.array([
        0.820905,
        0.884617,
        0.963558
    ])
}

# ------------------------------------------------------------
# estratégia IV
# ------------------------------------------------------------
estrategia_4 = {
    "bpp": np.array([
        0.160812,
        0.323918,
        1.021573
    ]),
    "ws_psnr": np.array([
        28.577817,
        30.876526,
        35.403960
    ]),
    "ws_ssim": np.array([
        0.815582,
        0.879992,
        0.955262
    ])
}

# ============================================================
# organização dos métodos
# ============================================================

metodos = [
    ("LALIC", lalic_pesos_pre_treinados),
    ("LALIC Baseline", lalic_baseline_30k),
    ("Estratégia I", estrategia_1),
    ("Estratégia II", estrategia_2),
    ("Estratégia III", estrategia_3),
    ("Estratégia IV", estrategia_4)
]

# ============================================================
# estilos dos métodos
# ============================================================

estilos = {
    "LALIC": {
        "linestyle": "-",
        "marker": "o"
    },
    "LALIC Baseline": {
        "linestyle": "--",
        "marker": "s"
    },
    "Estratégia I": {
        "linestyle": "-.",
        "marker": "^"
    },
    "Estratégia II": {
        "linestyle": ":",
        "marker": "D"
    },
    "Estratégia III": {
        "linestyle": "-",
        "marker": "P"
    },
    "Estratégia IV": {
        "linestyle": "--",
        "marker": "X"
    }
}

# ============================================================
# criação da figura
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(13.5, 5.8)
)

# ============================================================
# gráfico 1: bpp x ws-psnr
# ============================================================

ax = axes[0]

for nome, dados in metodos:

    ordem = np.argsort(dados["bpp"])

    bpp = dados["bpp"][ordem]
    ws_psnr = dados["ws_psnr"][ordem]

    estilo = estilos[nome]

    ax.plot(
        bpp,
        ws_psnr,
        linestyle=estilo["linestyle"],
        marker=estilo["marker"],
        markersize=7,
        linewidth=2,
        markeredgewidth=0.8,
        label=nome
    )

ax.set_xlabel("bpp")
ax.set_ylabel("WS-PSNR (dB)")
ax.set_title("(a) bpp × WS-PSNR")

ax.grid(
    True,
    linestyle="--",
    linewidth=0.8,
    alpha=0.35
)

ax.set_xlim(0.08, 1.10)
ax.set_ylim(27.5, 37.5)

# ------------------------------------------------------------
# inset da região de baixa taxa
# ------------------------------------------------------------

inset = inset_axes(
    ax,
    width="42%",
    height="42%",
    loc="lower right",
    borderpad=1.0
)

for nome, dados in metodos:

    ordem = np.argsort(dados["bpp"])

    bpp = dados["bpp"][ordem]
    ws_psnr = dados["ws_psnr"][ordem]

    estilo = estilos[nome]

    inset.plot(
        bpp,
        ws_psnr,
        linestyle=estilo["linestyle"],
        marker=estilo["marker"],
        markersize=4.5,
        linewidth=1.0,
        markeredgewidth=0.6
    )

inset.set_xlim(0.11, 0.37)
inset.set_ylim(27.8, 31.5)

inset.grid(
    True,
    linestyle="--",
    linewidth=0.5,
    alpha=0.35
)

inset.tick_params(
    labelsize=9
)

# ============================================================
# gráfico 2: bpp x ws-ssim
# ============================================================

ax = axes[1]

for nome, dados in metodos:

    ordem = np.argsort(dados["bpp"])

    bpp = dados["bpp"][ordem]
    ws_ssim = dados["ws_ssim"][ordem]

    estilo = estilos[nome]

    ax.plot(
        bpp,
        ws_ssim,
        linestyle=estilo["linestyle"],
        marker=estilo["marker"],
        markersize=7,
        linewidth=2,
        markeredgewidth=0.8,
        label=nome
    )

ax.set_xlabel("bpp")
ax.set_ylabel("WS-SSIM")
ax.set_title("(b) bpp × WS-SSIM")

ax.grid(
    True,
    linestyle="--",
    linewidth=0.8,
    alpha=0.35
)

ax.set_xlim(0.08, 1.10)
ax.set_ylim(0.775, 0.975)

# ------------------------------------------------------------
# inset da região de baixa taxa
# ------------------------------------------------------------

inset = inset_axes(
    ax,
    width="42%",
    height="42%",
    loc="lower right",
    borderpad=1.2
)

for nome, dados in metodos:

    ordem = np.argsort(dados["bpp"])

    bpp = dados["bpp"][ordem]
    ws_ssim = dados["ws_ssim"][ordem]

    estilo = estilos[nome]

    inset.plot(
        bpp,
        ws_ssim,
        linestyle=estilo["linestyle"],
        marker=estilo["marker"],
        markersize=4.5,
        linewidth=1.5,
        markeredgewidth=0.6
    )

inset.set_xlim(0.11, 0.37)
inset.set_ylim(0.775, 0.890)

inset.grid(
    True,
    linestyle="--",
    linewidth=0.5,
    alpha=0.35
)

inset.tick_params(
    labelsize=9
)

# ============================================================
# legenda única
# ============================================================

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=3,
    bbox_to_anchor=(0.5, -0.02),
    frameon=True,
    fontsize=11
)

# ============================================================
# ajustes finais
# ============================================================

plt.subplots_adjust(
    bottom=0.20,
    wspace=0.25
)

# ============================================================
# salvar figura
# ============================================================

plt.savefig(
    "rd_curves_ws2.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    "rd_curves_ws2.pdf",
    bbox_inches="tight"
)

plt.close()