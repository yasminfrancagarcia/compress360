
source .venv-lalic/bin/activate
Comando para compilar o compressor 


PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 \
python -u train360_wmse_crop.py \
    -d sun30 \
    --lambda 0.0025 \
    --epochs 20 \
    --lr_epoch 16 \
    --batch-size 3 \
    --num-workers 16 \
    --cuda \
    --save_path checkpoints360/check_w-mse_cropWIdthOnly_20epocas/check_0025/ \
    --save 2>&1 | tee checkpoints_30k_openImg/0.0025_train.log

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 \
python -u train360_wmse_crop.py \
    -d sun30 \
    --lambda 0.0483 \
    --epochs 20 \
    --lr_epoch 16 \
    --batch-size 3 \
    --num-workers 16 \
    --patch-size 512 256 \
    --cuda \
    --save_path checkpoints360/check_w-mse_cropWIdthOnly_20epocas/check_0483 \
    --save 2>&1 | tee checkpoints360/check_w-mse_cropWIdthOnly_20epocas/check_0483/00483_train.log

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 \
python -u train.py \
    -d openImages_30k_resized_1024x512 \
    --lambda 0.0483 \
    --epochs 20 \
    --lr_epoch 16 \
    --batch-size 4 \
    --num-workers 16 \
    --cuda \
    --save_path checkpoints_30k_openImg/check_0483 \
    --save 2>&1 | tee checkpoints_30k_openImg/0.0483_train.log

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 \
python -u train.py \
    -d sun30 \
    --lambda 0.0483 \
    --epochs 20 \
    --lr_epoch 16 \
    --batch-size 4 \
    --num-workers 16 \
    --cuda \
    --save_path checkpoints_30k_openImg/ \
    --save 2>&1 | tee checkpoints_30k_openImg/0.0483_train.log

 find ~/projeto/compress360/sun360test_500 -maxdepth 1 -type f | head -n 50 | xargs -I {} cp "{}" ~/projeto/compress360/amostras_50_sun360/

python eval.py \
    -m LALIC \
    -p checkpoints360/check_w-mse_cropWIdthOnly_20epocas/check_0067/0.0067checkpoint_best.pth.tar \
    -q 6 \
    -i sun360test_500 \
    -o recon_0067_wspsnrArrumado_ms_ssim500 \
    --result benchmark_0067_wspsnrArrumado_msssim500.json \
    --cuda \
    --real \
    --verbose



python eval.py \
    -m LALIC \
    -p checkpoints/lalic-q1.pth checkpoints/lalic-q2.pth checkpoints/lalic-q3.pth checkpoints/lalic-q4.pth checkpoints/lalic-q5.pth checkpoints/lalic-q6.pth \
    -q 1 2 3 4 5 6 \
    -i sun360test_500 \
    -o  recon_imagesSun360_test\
    --result benchmark_teste500Sun.json \
    --cuda \
    --real \
    --verbose

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 \
python train.py \
    -d sun30 \
    --lambda 0.0067 \
    --epochs 40 \
    --lr_epoch 36 \
    --batch-size 4 \
    --cuda \
    --save_path checkpoints_30k_openImg/ \
    --save 2>&1 | tee checkpoints_30k_openImg/0.0067_train.log

/////////////// to usando esse 
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 \
python -u train.py \
    -d sun30 \
    --lambda 0.0483 \
    --epochs 20 \
    --lr_epoch 16 \
    --batch-size 4 \
    --num-workers 12 \
    --patch-size 256 256 \
    --cuda \
    --save_path checkpoints360/check_sun30/check_0483_20epoc \
    --save \
    2>&1 | tee checkpoints360/check_sun30/0.0483_train.log

treino com o train normal 


 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 python train.py \
    -d sun30 \
    --lambda 0.0067 \
    --epochs 20 \
    --lr_epoch 16 \
    --batch-size 4 \
    --patch-size 256 256 \
    --cuda \
    --save_path checkpoints360/check_sun30/check_0067_20epoc \
    --save
    2>&1 | tee checkpoints360/check_sun30/check_0067_20epoc/0.0067_20epoc_train.log

////////////////////////////////////////////////////////////////////////////////////////////



PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 python eval.py \
    -m LALIC \
    -p checkpoints360/check_w-mse_randomCrop_doZero/0.0067checkpoint_best.pth.tar\
    -q 3 \
    -i sun360test_500 \
    -o recon_test500_w-mse_randomCrop_20epoc \
    --result benchmark_test500_0067_wmse_cropRand_20epoc_doZero.json \
    --cuda \
    --real \
    --verbose



////////////////// treino atual deepcool: com w-mse + random crop

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 python -u train360.py \
    -d sun30 \
    --lambda 0.0025 \
    --epochs 20\
    --lr_epoch 16 \
    --batch-size 4 \
    --patch-size 256 256 \
    --num-workers 12 \
    --cuda \
    --save_path checkpoints360/check_w-mse_randomCrop_doZero/check_0025 \
    --save \
    2>&1 | tee checkpoints360/check_w-mse_randomCrop_doZero/check_0025/0025_train360_mse_doZero.log

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 python -u train360.py \
    -d sun30 \
    --lambda 0.0483 \
    --epochs 20\
    --lr_epoch 16 \
    --batch-size 4 \
    --patch-size 256 256 \
    --num-workers 12 \
    --cuda \
    --save_path checkpoints360/check_w-mse_randomCrop_doZero/check_0483 \
    --save \
    2>&1 | tee checkpoints360/check_w-mse_randomCrop_doZero/check_0483/0483_train360_mse_doZero.log


//////////////////////////////// teste com convoluções adptadas 


PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 python -u train_swhdc.py \
    -d sun30 \
    --lambda 0.0025 \
    --epochs 20 \
    --lr_epoch 16 \
    --batch-size 3 \
    --patch-size 512 256 \
    --num-workers 16 \
    --swhdc_tag \
    --swhdc_dilations 1 2 3 \
    --cuda \
    --save_path checkpoints360/check_swhdc_certo/check_1 \
    --save \
    2>&1 |tee checkpoints360/check_swhdc_certo/check_1/train_swhdc_certo.log

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 python -u train_swhdc.py \
    -d sun30 \
    --lambda 0.0067 \
    --epochs 20 \
    --lr_epoch 16 \
    --batch-size 3 \
    --patch-size 512 256 \
    --num-workers 16 \
    --swhdc_tag \
    --swhdc_dilations 1 2 3 \
    --cuda \
    --save_path checkpoints360/check_swhdc_certo/check_2 \
    --save \
    2>&1 |tee checkpoints360/check_swhdc_certo/check_2/train_swhdc_certo.log


//pra ver a época do latest
    python - <<'PY'
import torch

path = "0.0483checkpoint_latest.pth.tar"

ckpt = torch.load(path, map_location="cpu", weights_only=False)

print("Tipo:", type(ckpt))

if isinstance(ckpt, dict):
    print("\nChaves:")
    for k, v in ckpt.items():
        print(f"  {k}: {v}")
PY

//pra ver qual a best epoch
python - <<'PY'
import torch

for path in [
    "0.0025checkpoint_latest.pth.tar",
    "0.0025checkpoint_best.pth.tar",
]:
    print("\n====================")
    print(path)

    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict):
        for k, v in ckpt.items():
            if not isinstance(v, dict):
                print(k, "=", v)
PY

