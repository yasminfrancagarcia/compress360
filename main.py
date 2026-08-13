from PIL import Image

import os
sample = os.listdir("sun360/train")[:300]  # ajuste o caminho

caminho = "sun360/train" 
total = sum(1 for arquivo in os.listdir(caminho) if arquivo.endswith(".jpg") or arquivo.endswith(".png"))
print(f"Total de imagens: {total}")