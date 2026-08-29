import os
import shutil
import random
import fiftyone as fo
import fiftyone.zoo as foz
from PIL import Image

# Lista expandida com 45 classes para ter margem de sobra
classes_desejadas = [
    # Outdoor / Natureza
    "Tree", "Bench", "Fountain", "Flower", "Palm tree", "Plant", "Grass", "Mountain",
    # Arquitetura / Estruturas
    "House", "Building", "Porch", "Tower", "Stairs", "Window", "Door", "Bridge", "Skyscraper",
    # Indoor / Móveis / Casa
    "Bed", "Couch", "Kitchen appliance", "Chair", "Table", "Cabinetry", "Shelf", "Sink", "Countertop", "Mirror",
    # Tecnologia / Laboratório / Objetos
    "Beaker", "Medical equipment", "Computer monitor", "Clock", "Bookcase", "Laptop", "Television",
    # Veículos / Transporte
    "Car", "Airplane", "Bus", "Watercraft", "Bicycle", "Motorcycle", "Truck", "Train", "Van", "Helicopter"
]

output_dir = "./dataset_open_images_30k/test"
os.makedirs(output_dir, exist_ok=True)

META_TOTAL = 1500
MAX_POR_CLASSE = 30

total_geral = 0

print(f"Buscando imagens APENAS no split 'validation'...")
print(f"Meta global: EXATAMENTE {META_TOTAL} imagens (máximo de {MAX_POR_CLASSE} por classe)\n")

for classe in classes_desejadas:
    if total_geral >= META_TOTAL:
        print(f"\n META GLOBAL ATINGIDA! Total de {total_geral} imagens coletadas.")
        break

    aprovadas_classe = 0
    tentativa = 1
    
    print(f"---> Coletando classe: {classe} (Total acumulado: {total_geral}/{META_TOTAL})")

    # Tenta até 2 vezes buscar amostras para a classe atual
    while aprovadas_classe < MAX_POR_CLASSE and tentativa <= 2:
        dataset_name_unico = f"temp_val_{classe.lower().replace(' ', '_')}_{tentativa}_{random.randint(1000, 9999)}"
        
        try:
            dataset = foz.load_zoo_dataset(
                "open-images-v7",
                split="validation",
                label_types=["detections"],
                classes=[classe],
                max_samples=800,
                shuffle=True,
                only_matching=True,
                dataset_name=dataset_name_unico
            )

            novas_nesta_rodada = 0
            for sample in dataset:
                # Interrompe se atingiu a cota da classe ou a meta global de 1500
                if aprovadas_classe >= MAX_POR_CLASSE or total_geral >= META_TOTAL:
                    break

                src_path = sample.filepath
                img_name = os.path.basename(src_path)
                dst_path = os.path.join(output_dir, img_name)

                try:
                    with Image.open(src_path) as img:
                        w, h = img.size
                        # Filtro de resolução mínima
                        if w >= 1024 and h >= 512:
                            if not os.path.exists(dst_path):
                                shutil.copy(src_path, dst_path)
                                aprovadas_classe += 1
                                total_geral += 1
                                novas_nesta_rodada += 1
                except Exception:
                    pass

            # Limpa o dataset temporário do FiftyOne
            if fo.dataset_exists(dataset_name_unico):
                dataset.delete()

            # SE NÃO ENCONTROU NENHUMA NOVA IMAGEM NESTA RODADA:
            # Significa que as imagens grandes dessa classe no 'validation' acabaram.
            if novas_nesta_rodada == 0:
                print(f"   Sem mais imagens >= 1024x512 para '{classe}'. Encerrando classe com {aprovadas_classe} fotos.")
                break

            tentativa += 1

        except Exception as e:
            print(f"  Erro ao processar classe {classe}: {e}")
            break

    print(f"  Resultado da classe {classe}: {aprovadas_classe} imagens aprovadas.\n")

print(f"Processo finalizado!")
print(f"Total de imagens salvas em {output_dir}: {total_geral}")