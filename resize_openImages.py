import os
from PIL import Image


pasta_origem = "./dataset_open_images_30k/test2"   
pasta_destino = "./dataset_resized_1024x512/test"  

# na biblioteca PIL, a ordem das dimensões é (largura, altura)
TAMANHO_DESEJADO = (1024, 512)

os.makedirs(pasta_destino, exist_ok=True)

# Extensões válidas de imagem
extensoes_validas = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

arquivos = [f for f in os.listdir(pasta_origem) if f.lower().endswith(extensoes_validas)]
total_arquivos = len(arquivos)

print(f"iniciando redimensionamento de {total_arquivos} imagens para 1024x512...")

processadas = 0
erros = 0

for i, nome_arquivo in enumerate(arquivos, start=1):
    caminho_origem = os.path.join(pasta_origem, nome_arquivo)
    caminho_destino = os.path.join(pasta_destino, nome_arquivo)

    try:
        with Image.open(caminho_origem) as img:
            # Redimensiona para 1024 de largura por 512 de altura
            # Resampling.LANCZOS garante a melhor qualidade visual na interpolação
            img_resized = img.resize(TAMANHO_DESEJADO, Image.Resampling.LANCZOS)
            
            # Converte para RGB caso haja imagens PNG com canal alpha (transparência)
            if img_resized.mode in ("RGBA", "P"):
                img_resized = img_resized.convert("RGB")
                
            img_resized.save(caminho_destino, quality=95)
            processadas += 1

    except Exception as e:
        print(f"Erro ao processar {nome_arquivo}: {e}")
        erros += 1

    # exibe progresso a cada 1000 imagens
    if i % 1000 == 0 or i == total_arquivos:
        print(f"Progresso: {i}/{total_arquivos} imagens processadas.")

print(f"\nConcluído!")
print(f"Imagens redimensionadas com sucesso: {processadas}")
print(f"Falhas: {erros}")
print(f"Salvas em: {pasta_destino}")