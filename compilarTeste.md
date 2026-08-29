Lalic original (treinado com 40 épocas e 400 mil imagens planas) no data set de 50 amostras 

python eval.py \
    -m LALIC \
    -p checkpoints_originais/lalic-q1.pth checkpoints_originais/lalic-q3.pth checkpoints_originais/lalic-q6.pth \
    -q 1 3 6 \
    -i amostras_50_sun360 \
    -o recon_images_50amostras/lalic_original3q \
    --result benchmarks_50amostras/benchmark_3q_original.json \
    --cuda \
    --real \
    --verbose


python eval.py \
    -m LALIC \
    -p checkpoints_original_train_on_360/0.0025checkpoint_best.pth.tar checkpoints_original_train_on_360/0.0067checkpoint_best.pth.tar checkpoints_original_train_on_360/0.0483checkpoint_best.pth.tar \
    -q 1 3 6 \
    -i amostras_50_sun360 \
    -o recon_images_50amostras/lalic_original3q \
    --result benchmarks_50amostras/benchmark_3q_original_train360.json \
    --cuda \
    --real \
    --verbose


teste com 3 qualidades de w-mse + crop width : 

python eval.py \
    -m LALIC \
    -p checkpoints_wmse_crop_360/check_w-mse_cropWIdthOnly_20epocas/check_0025/0.0025checkpoint_best.pth.tar checkpoints_wmse_crop_360/check_w-mse_cropWIdthOnly_20epocas/check_0067/0.0067checkpoint_best.pth.tar checkpoints_wmse_crop_360/check_w-mse_cropWIdthOnly_20epocas/check_0483/0.0483checkpoint_best.pth.tar  \
    -q 1 3 6 \
    -i amostras_50_sun360 \
    -o recon_images_50amostras/wmse_cropWidth_3q \
    --result benchmarks_50amostras/benchmark_3q_wmse_cropwidth.json \
    --cuda \
    --real \
    --verbose


teste com lambda 0,0067 (qualidade 3), lalic original, treinado por 40 epocas em 30k de imagens 360 ((MSE normal e random crop fixo de 256x256)
)

agora é o random crop + wmse
python eval.py \
    -m LALIC \
    -p randomCrop_wmse/0.0067checkpoint_best.pth.tar  \
    -q 3 \
    -i amostras_50_sun360 \
    -o recon_images_50amostras/wmse_randomCrop_1q \
    --result benchmarks_50amostras/benchmark_wmse_randomCrop.json \
    --cuda \
    --real \
    --verbose

teste com as 50 imagens e  o dataset treinado com 20 epocas em 30k imagens planas

teste com as 3 qualidades treinado com 30k imagens planas
python eval.py \
    -m LALIC \
    -p checkpoints_30k_openImg/check_0025/0.0025checkpoint_best.pth.tar checkpoints_30k_openImg/check_0067/0.0067checkpoint_best.pth.tar checkpoints_30k_openImg/check_0483/0.0483checkpoint_best.pth.tar\
    -q 1 3 6 \
    -i amostras_50_sun360 \
    -o recon_images_50amostras/train_on_30kOpenI \
    --result benchmarks_50amostras/benchmark_train_on_30kOpenI_3q.json \
    --cuda \
    --real \
    --verbose