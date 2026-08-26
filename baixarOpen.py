import os
import shutil
import fiftyone.zoo as foz
from PIL import Image

classes_desejadas = [
    "Tree", "Bench", "Fountain",
    "House", "Building", "Porch",
    "Bed", "Couch", "Window", "Kitchen appliance",
    "Beaker", "Medical equipment", "Computer monitor",
    "Car", "Airplane", "Bus"
]

output_dir = "./dataset_open_images_30k/test"
os.makedirs(output_dir, exist_ok=True)

META_TOTAL_IMAGENS = 1500

print(f"Baixando dataset global para filtrar imagens >= 1024x512...")

# Baixa um pool maior de imagens DE UMA SÓ VEZ para carregar metadados Apenas 1 Vez
# Solicitamos 60.000 amostras misturadas para garantir que 30.000 passem no filtro de tamanho
dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="validation",
    label_types=["detections"],
    classes=classes_desejadas,
    max_samples=4000,
    shuffle=True,
    only_matching=True,
    seed=101
)

print("\nProcessando e filtrando imagens por resolução (>= 1024x512)...")

imagens_aprovadas = 0

# Iteramos sobre o dataset baixado pelo FiftyOne
for sample in dataset:
    if imagens_aprovadas >= META_TOTAL_IMAGENS:
        break

    src_path = sample.filepath
    img_name = os.path.basename(src_path)
    dst_path = os.path.join(output_dir, img_name)

    try:
        with Image.open(src_path) as img:
            w, h = img.size
            if w >= 1024 and h >= 512:
                if not os.path.exists(dst_path):
                    shutil.copy(src_path, dst_path)
                    imagens_aprovadas += 1
    except Exception:
        pass

print(f"\nFinalizado! Total de {imagens_aprovadas} imagens aprovadas em: {output_dir}")