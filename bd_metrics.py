import numpy as np
import bjontegaard as bd


# ============================================================
# DADOS
# ============================================================

# ------------------------------------------------------------
# 1. LALIC original
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
# 2. LALIC 30k 360 + MSE
# ------------------------------------------------------------
lalic_360_mse = {
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
# 3. LALIC 30k planas + MSE
# ------------------------------------------------------------
lalic_30k_planas = {
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

# 4. LALIC 30k 360 + W-MSE + random crop 256 256 

lalic_360_wmse_Randomcrop = {
    "bpp": [
      0.166709,
      0.338719,
      1.06543
    ],
    "ws_psnr": [
      28.581836,
      30.90475,
      36.497981

    ]
}


# ------------------------------------------------------------
# 4. LALIC 30k 360 + W-MSE + CropWidth
# ------------------------------------------------------------
lalic_wmse_cropwidth = {
    "bpp": np.array([
        0.168745,
        0.335549,
        1.029863
    ]),
    "ws_psnr": np.array([
        28.778219,
        31.197338,
        36.995555
    ])
}


# ============================================================
# FUNÇÃO PARA CALCULAR BD-RATE E BD-WS-PSNR
# ============================================================

def calcular_bd(nome_ref, ref, nome_test, test):

    #BD-Rate
    bd_rate = bd.bd_rate(
        ref["bpp"],
        ref["ws_psnr"],
        test["bpp"],
        test["ws_psnr"],
        method="pchip"
    )

    #BD-WS-PSNR
    bd_wspsnr = bd.bd_psnr(
        ref["bpp"],
        ref["ws_psnr"],
        test["bpp"],
        test["ws_psnr"],
        method="pchip"
    )

    print("=" * 5)
    print(f"Referência: {nome_ref}")
    print(f"Comparado:  {nome_test}")
    print("=" * 5)
    print(f"BD-Rate:    {bd_rate:.2f}%")
    print(f"BD-WS-PSNR: {bd_wspsnr:.4f} dB")
    print("=" * 5)

    return bd_rate, bd_wspsnr

# w-mse + 360 + random crop 256 256 vs lalic original 

# calcular_bd(
#     "LALIC original",
#     lalic_original,
#     "LALIC 30k 360 + W-MSE + RandomCrop",
#     lalic_360_wmse_Randomcrop
# )

# # w-mse + 360 + random crop 256 256 vs lalic 360 + mse

# calcular_bd(
#     "LALIC 30k 360 + MSE",
#     lalic_360_mse,
#     "LALIC 30k 360 + W-MSE + RandomCrop",
#     lalic_360_wmse_Randomcrop
# )


# # 360 + MSE vs W-MSE + CropWidth

# calcular_bd(
#     "LALIC 30k 360 + MSE",
#     lalic_360_mse,
#     "LALIC 30k 360 + W-MSE + CropWidth",
#     lalic_wmse_cropwidth
# )

# # wmse + 360 + random crop 256 256 vs lalic 360 + wmse + cropwidth

# calcular_bd(
#     "LALIC 30k 360 + W-MSE + RandomCrop",
#     lalic_360_wmse_Randomcrop,
#     "LALIC 30k 360 + W-MSE + CropWidth",
#     lalic_wmse_cropwidth
# )

# # W-MSE + CropWidth vs LALIC original

# calcular_bd(
#     "LALIC original",
#     lalic_original,
#     "LALIC 30k 360 + W-MSE + CropWidth",
#     lalic_wmse_cropwidth
# )


# # Treino com imagens 360 + MSE vs treino com imagens planas

# calcular_bd(
#     "LALIC 30k planas + MSE",
#     lalic_30k_planas,
#     "LALIC 30k 360 + MSE",
#     lalic_360_mse
# )


# # Método proposto vs treino com 30k imagens planas

# calcular_bd(
#     "LALIC 30k planas + MSE",
#     lalic_30k_planas,
#     "LALIC 30k 360 + W-MSE + CropWidth",
#     lalic_wmse_cropwidth
# )

calcular_bd(
    "LALIC original",
    lalic_original,
    "30k planas + MSE",
    lalic_30k_planas
)

calcular_bd(
    "LALIC original",
    lalic_original,
    "30k 360 + MSE",
    lalic_360_mse
)

calcular_bd(
    "LALIC original",
    lalic_original,
    "360 + W-MSE + Random Crop",
    lalic_360_wmse_Randomcrop
)

calcular_bd(
    "LALIC original",
    lalic_original,
    "360 + W-MSE + Crop Width",
    lalic_wmse_cropwidth
)

# ---------------------------------------------------------

# compaaração com o original treinado em 30k planas por 20 épocas 


calcular_bd(
    "LALIC original treinado com 30k planas (1024x512), por 20 épocas",
    lalic_30k_planas,
    "30k 360 + MSE",
    lalic_360_mse
)

calcular_bd(
    "LALIC original treinado com 30k planas (1024x512), por 20 épocas",
    lalic_30k_planas,
    "360 + W-MSE + Random Crop",
    lalic_360_wmse_Randomcrop
)

calcular_bd(
    "LALIC original treinado com 30k planas (1024x512), por 20 épocas   ",
    lalic_30k_planas,
    "360 + W-MSE + Crop Width",
    lalic_wmse_cropwidth
)

def plot_rcd_vs_anchor(nome_anchor, anchor, nome_testes):

    for nome_test, test in nome_testes.items():

        print("=" * 10)
        print(f"Anchor: {nome_anchor}")
        print(f"Test:   {nome_test}")
        print("=" * 10)

        bd.plot_rcd(
            anchor["bpp"],
            anchor["ws_psnr"],
            test["bpp"],
            test["ws_psnr"],
            method="pchip",
            require_matching_points=True,
            samples=1000
        )

testes_vs_original = {
    "30k planas + MSE": lalic_30k_planas,

    "30k 360 + MSE": lalic_360_mse,

    "30k 360 + W-MSE + Random Crop":
        lalic_360_wmse_Randomcrop,

    "30k 360 + W-MSE + Crop Width":
        lalic_wmse_cropwidth
}

testes_vs_30k_planas = {
    "30k 360 + MSE": lalic_360_mse,

    "30k 360 + W-MSE + Random Crop":
        lalic_360_wmse_Randomcrop,

    "30k 360 + W-MSE + Crop Width":
        lalic_wmse_cropwidth
}



plot_rcd_vs_anchor(
    "LALIC original",
    lalic_original,
    testes_vs_original
)

plot_rcd_vs_anchor(
    "30k planas + MSE",
    lalic_30k_planas,
    testes_vs_30k_planas
)