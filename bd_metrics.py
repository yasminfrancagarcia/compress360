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

    print("=" * 70)
    print(f"Referência: {nome_ref}")
    print(f"Comparado:  {nome_test}")
    print("-" * 70)
    print(f"BD-Rate:    {bd_rate:.2f}%")
    print(f"BD-WS-PSNR: {bd_wspsnr:.4f} dB")
    print("=" * 70)

    return bd_rate, bd_wspsnr



# 360 + MSE vs W-MSE + CropWidth

calcular_bd(
    "LALIC 30k 360 + MSE",
    lalic_360_mse,
    "LALIC 30k 360 + W-MSE + CropWidth",
    lalic_wmse_cropwidth
)


# W-MSE + CropWidth vs LALIC original

calcular_bd(
    "LALIC original",
    lalic_original,
    "LALIC 30k 360 + W-MSE + CropWidth",
    lalic_wmse_cropwidth
)


# Treino com imagens 360 + MSE vs treino com imagens planas

calcular_bd(
    "LALIC 30k planas + MSE",
    lalic_30k_planas,
    "LALIC 30k 360 + MSE",
    lalic_360_mse
)


# Método proposto vs treino com 30k imagens planas

calcular_bd(
    "LALIC 30k planas + MSE",
    lalic_30k_planas,
    "LALIC 30k 360 + W-MSE + CropWidth",
    lalic_wmse_cropwidth
)