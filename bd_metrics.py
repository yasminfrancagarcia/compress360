import numpy as np
import bjontegaard as bd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"


plt.rcParams["font.size"] = 14
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["legend.fontsize"] = 12


# ------------------------------------------------------------
# lalic original
# ------------------------------------------------------------
lalic_original = {
    "bpp": np.array([
        0.128777,
        0.328257,
        0.938292
    ]),
    "ws_psnr": np.array([
        27.972017,
        31.224781,
        36.833692
    ])
}

# ------------------------------------------------------------
# lalic 30k planas + mse
# ------------------------------------------------------------
lalic_baseline = {
    "bpp": np.array([
        0.158131,
        0.326836,
        1.025072
    ]),
    "ws_psnr": np.array([
        28.139117,
        30.424208,
        35.586628
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
    ])
}

# ============================================================
# função para calcular bd-rate e bd-ws-psnr
# ============================================================

def calcular_bd(nome_ref, ref, nome_test, test):

    bd_rate = bd.bd_rate(
        ref["bpp"],
        ref["ws_psnr"],
        test["bpp"],
        test["ws_psnr"],
        method="pchip"
    )

    bd_wspsnr = bd.bd_psnr(
        ref["bpp"],
        ref["ws_psnr"],
        test["bpp"],
        test["ws_psnr"],
        method="pchip"
    )

    print("=" * 60)
    print(f"referência: {nome_ref}")
    print(f"comparado:  {nome_test}")
    print("=" * 60)
    print(f"bd-rate:    {bd_rate:.2f}%")
    print(f"bd-ws-psnr: {bd_wspsnr:.4f} db")
    print("=" * 60)

    return bd_rate, bd_wspsnr


# ============================================================
# comparações com o lalic original
# ============================================================

print("\n")
print("#" * 60)
print("comparações com o lalic original")
print("#" * 60)

calcular_bd(
    "lalic",
    lalic_original,
    "30k planas + mse",
    lalic_baseline
)

calcular_bd(
    "lalic",
    lalic_original,
    "estratégia I",
    estrategia_1
)

calcular_bd(
    "lalic",
    lalic_original,
    "estratégia II",
    estrategia_2
)

calcular_bd(
    "lalic",
    lalic_original,
    "estratégia III",
    estrategia_3
)

calcular_bd(
    "lalic",
    lalic_original,
    "estratégia IV",
    estrategia_4
)

# ============================================================
# comparações com o baseline
# ============================================================

print("\n")
print("#" * 60)
print("comparações com o baseline")
print("#" * 60)

calcular_bd(
    "30k planas + mse",
    lalic_baseline,
    "estratégia I",
    estrategia_1
)

calcular_bd(
    "30k planas + mse",
    lalic_baseline,
    "estratégia II",
    estrategia_2
)

calcular_bd(
    "30k planas + mse",
    lalic_baseline,
    "estratégia III",
    estrategia_3
)

calcular_bd(
    "30k planas + mse",
    lalic_baseline,
    "estratégia IV",
    estrategia_4
)

# ============================================================
# função para gerar rcd
# ============================================================

def plot_rcd_vs_anchor(nome_anchor, anchor, nome_testes, nome_arquivo):

    for nome_test, test in nome_testes.items():

        print("\n" + "=" * 60)
        print(f"anchor: {nome_anchor}")
        print(f"test:   {nome_test}")
        print("=" * 60)

        # gerar o rcd
        bd.plot_rcd(
            anchor["bpp"],
            anchor["ws_psnr"],
            test["bpp"],
            test["ws_psnr"],
            method="pchip",
            require_matching_points=True,
            samples=1000
        )

        # ajustar layout
        plt.tight_layout()

        # salvar a figura
        plt.savefig(
            nome_arquivo(nome_test),
            dpi=300,
            bbox_inches="tight"
        )

        # fechar a figura para evitar sobreposição
        plt.close()


# ============================================================
# testes contra o lalic original
# ============================================================

testes_vs_original = {
    "30k planas + MSE": lalic_baseline,
    "Estratégia I": estrategia_1,
    "Estratégia II": estrategia_2,
    "Estratégia III": estrategia_3,
    "Estratégia IV": estrategia_4
}

# ============================================================
# testes contra o baseline
# ============================================================

testes_vs_30k_planas = {
    "Estratégia I": estrategia_1,
    "Estratégia II": estrategia_2,
    "Estratégia III": estrategia_3,
    "Estratégia IV": estrategia_4
}

# ============================================================
# nomes dos arquivos rcd
# ============================================================

def nome_rcd_original(nome):
    nome = nome.lower()
    nome = nome.replace(" ", "_")
    nome = nome.replace("+", "")
    return f"rcd_vs_lalic_original_{nome}.pdf"


def nome_rcd_baseline(nome):
    nome = nome.lower()
    nome = nome.replace(" ", "_")
    return f"rcd_vs_baseline_{nome}.pdf"


# ============================================================
# gerar rcd contra lalic original
# ============================================================

plot_rcd_vs_anchor(
    "lalic original",
    lalic_original,
    testes_vs_original,
    nome_rcd_original
)

# ============================================================
# gerar rcd contra baseline
# ============================================================

plot_rcd_vs_anchor(
    "30k planas + mse",
    lalic_baseline,
    testes_vs_30k_planas,
    nome_rcd_baseline
)

# ============================================================
# compare methods
# ============================================================

# ------------------------------------------------------------
# compare methods: lalic original x estratégia III
# ------------------------------------------------------------

bd.compare_methods(
    lalic_original["bpp"],
    lalic_original["ws_psnr"],
    estrategia_3["bpp"],
    estrategia_3["ws_psnr"],
    rate_label="bpp",
    distortion_label="ws-psnr (db)",
    figure_label="lalic vs estratégia iii",
    filepath="compare_methods_lalic_vs_estrategia_iii.pdf"
)

# ------------------------------------------------------------
# compare methods: baseline x estratégia III
# ------------------------------------------------------------

bd.compare_methods(
    lalic_baseline["bpp"],
    lalic_baseline["ws_psnr"],
    estrategia_3["bpp"],
    estrategia_3["ws_psnr"],
    rate_label="bpp",
    distortion_label="ws-psnr (db)",
    figure_label="baseline vs estratégia iii",
    filepath="compare_methods_baseline_vs_estrategia_iii.pdf"
)

print("\n")
print("=" * 60)
print("arquivos gerados com sucesso")
print("=" * 60)
print("rcd_vs_lalic_original_*.pdf")
print("rcd_vs_baseline_*.pdf")
print("compare_methods_lalic_vs_estrategia_iii.pdf")
print("compare_methods_baseline_vs_estrategia_iii.pdf")
print("=" * 60)

