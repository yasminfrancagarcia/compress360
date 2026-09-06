from pathlib import Path
import zipfile
import shutil
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12
# ============================================================
# CONFIGURAÇÃO
# ============================================================
CROP_X = 190
CROP_Y = 310
CROP_SIZE_x = 80
CROP_SIZE_y = 100
# Ordem desejada na figura
ORDER = [
    "lalic baseline",
    "estratégia 1",
    "estratégia 2",
    "estratégia 3",
    "estratégia 4",
]

LABELS = {
    "lalic baseline": "LALIC Baseline",
    "estratégia 1": "Estratégia 1",
    "estratégia 2": "Estratégia 2",
    "estratégia 3": "Estratégia 3",
    "estratégia 4": "Estratégia 4",
}

def norm(s):
    return s.strip().lower()

def locate_dataset(root: Path):
    # encontra a pasta que contém as subpastas do experimento
    candidates = [root] + [p for p in root.rglob("*") if p.is_dir()]

    for p in candidates:
        names = {norm(x.name) for x in p.iterdir() if x.is_dir()}

        if "imagem_original" in names and "lalic baseline" in names:
            return p

    raise RuntimeError(
        "Não encontrei a pasta principal do conjunto de reconstruções."
    )

def load_images(base: Path):
    images = []

    # ========================================================
    # IMAGEM ORIGINAL
    # ========================================================
    orig_dir = next(
        p for p in base.iterdir()
        if p.is_dir() and norm(p.name) == "imagem_original"
    )

    orig_files = [
        p for p in orig_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    ]

    if not orig_files:
        raise RuntimeError("Imagem original não encontrada.")

    original = Image.open(orig_files[0]).convert("RGB")
    images.append(("Original", original))

    # ========================================================
    # RECONSTRUÇÕES q6
    # ========================================================
    folders = {
        norm(p.name): p
        for p in base.iterdir()
        if p.is_dir()
    }

    for folder_name in ORDER:

        if folder_name not in folders:
            print(
                f"[aviso] pasta não encontrada: {folder_name}"
            )
            continue

        # ignora explicitamente lalic_original
        if "lalic_original" in folder_name.replace(" ", "_"):
            continue

        folder = folders[folder_name]

        q6_candidates = [
            p for p in folder.iterdir()
            if p.is_file()
            and p.stem.lower() == "q6"
            and p.suffix.lower() in (".jpg", ".jpeg", ".png")
        ]

        if not q6_candidates:
            print(
                f"[aviso] q6 não encontrado em: {folder.name}"
            )
            continue

        img = Image.open(q6_candidates[0]).convert("RGB")

        if img.size != original.size:
            raise RuntimeError(
                f"Tamanho diferente em {folder.name}: "
                f"{img.size} vs original {original.size}"
            )

        images.append(
            (
                LABELS.get(folder_name, folder.name),
                img
            )
        )

    return images

def make_crop_comparison(
    images,
    output_path: Path,
    x=CROP_X,
    y=CROP_Y,
    size_x=CROP_SIZE_x,
    size_y=CROP_SIZE_y
):

    W, H = images[0][1].size

    if (
        x < 0
        or y < 0
        or x + size_x > W
        or y + size_y > H
    ):
        raise ValueError(
            f"Crop ({x},{y},{size_x},{size_y}) "
            f"está fora da imagem {W}x{H}."
        )

    n = len(images)

    # ========================================================
    # CONTEXTO + RETÂNGULO
    # ========================================================
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.imshow(images[0][1])

    rect = patches.Rectangle(
        (x, y),
        size_x,
        size_y,
        linewidth=2,
        edgecolor="red",
        facecolor="none"
    )

    ax.add_patch(rect)

    ax.set_title(
        f"Localização do crop {size_x}×{size_y}"
    )

    ax.axis("off")

    fig.tight_layout()

    context_path = output_path.with_name(
        output_path.stem + "_contexto.png"
    )

    fig.savefig(
        context_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    # ========================================================
    # COMPARAÇÃO DOS CROPS
    # ========================================================
    fig, axes = plt.subplots(
        1,
        n,
        figsize=(1.5 * n, 2.0),
        squeeze=False
    )

    
    axes = axes[0]

    for ax, (title, img) in zip(axes, images):

        crop = img.crop(
            (
                x,
                y,
                x + size_x,
                y + size_y
            )
        )

        ax.imshow(
            crop,
            interpolation="nearest"
        )

        ax.set_title(
            title,
            fontsize=10
        )

        ax.axis("off")

    """ fig.suptitle(
        f"Comparação visual — qualidade q6 — "
        f"crop {size_x}×{size_y} em ({x},{y})",
        fontsize=13
    ) """

    fig.subplots_adjust(
        wspace=0.05
    )

   

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"Figura salva em: {output_path}"
    )

    print(
        f"Contexto salvo em: {context_path}"
    )

    print(
        f"Crop usado: "
        f"x={x}, y={y}, "
        f"tamanho={size_x}x{size_y}"
    )

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Compara crops  "
            "das reconstruções q6."
        )
    )

    parser.add_argument(
        "input",
        help=(
            "Caminho para o .zip "
            "ou para a pasta já extraída."
        )
    )

    parser.add_argument(
        "--output",
        default="comparacao2_crops_q6.png",
        help="Nome da figura de saída."
    )

    parser.add_argument(
        "--x",
        type=int,
        default=CROP_X
    )

    parser.add_argument(
        "--y",
        type=int,
        default=CROP_Y
    )

    parser.add_argument(
    "--size_x",
    type=int,
    default=CROP_SIZE_x
    )

    parser.add_argument(
        "--size_y",
        type=int,
        default=CROP_SIZE_y
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    temp_dir = None

    # ========================================================
    # SE FOR ZIP, EXTRAI TEMPORARIAMENTE
    # ========================================================
    if input_path.suffix.lower() == ".zip":

        temp_dir = Path(
            "_tmp_recon_extract"
        )

        if temp_dir.exists():
            shutil.rmtree(temp_dir)

        temp_dir.mkdir(
            parents=True
        )

        with zipfile.ZipFile(
            input_path,
            "r"
        ) as z:

            z.extractall(
                temp_dir
            )

        root = temp_dir

    else:
        root = input_path

    try:

        base = locate_dataset(root)

        images = load_images(base)

        make_crop_comparison(
                images,
                Path(args.output),
                x=args.x,
                y=args.y,
                size_x=args.size_x,
                size_y=args.size_y
            )

    finally:

        if (
            temp_dir
            and temp_dir.exists()
        ):
            shutil.rmtree(
                temp_dir
            )

if __name__ == "__main__":
    main()