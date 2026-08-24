
source .venv-lalic/bin/activate
Comando para compilar o compressor 



for lmbda in 0.0018 0.0067 0.0130 0.0250 0.0483; do
    CUDA_VISIBLE_DEVICES=0 python train.py \
        -d /path/to/dataset \
        --lambda $lmbda \
        --epochs 40 \
        --lr_epoch 36 \
        --batch-size 8 \
        --save_path /path/for/saving --save
done



 find ~/projeto/RwkvCompress360/sun360/test -maxdepth 1 -type f | head -n 500 | xargs -I {} cp "{}" ~/projeto/RwkvCompress360/sun360test_500/

python eval.py \
    -m LALIC \
    -p checkpoints360/0.00483checkpoint_best.pth.tar \
    -q 6 \
    -i sun360/amostras50 \
    -o recon_images_048-5epoc \
    --result benchmark_amostras048_5epoc.json \
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
    --save_path checkpoints360/check_sun30/check_40epoc \
    --save 2>&1 | tee checkpoints360/check_40epoc/0.0067_train.log

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

//fine tuning com a epoca 15
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 python train.py \
    -d sun30 \
    --lambda 0.0067 \
    --checkpoint checkpoints360/check_sun30/check_0067_40epoc/0.006715_checkpoint.pth.tar \
    --epochs 20 \
    --lr_epoch 16
    --batch-size 4 \
    --patch-size 256 256 \
    --num-workers 12 \
    --cuda \
    --save_path checkpoints360/check_0067_20epoc \
    --save

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 python train.py \
    -d sun30 \
    --lambda 0.0067 \
    --checkpoint checkpoints360/check_sun30/check_0067_40epoc/0.006715_checkpoint.pth.tar \
    --epochs 20 \
    --learning-rate 0.0001 \
    --lr_epoch 16 \
    --batch-size 4 \
    --patch-size 256 256 \
    --num-workers 12 \
    --cuda \
    --save_path checkpoints360/check_sun30/check_0067_20epoc \
    --save
    2>&1 | tee checkpoints360/check_sun30/check_0067_20epoc/0.0067_20epoc_train.log

python eval.py \
    -m LALIC \
    -p checkpoints360/check_sun30/check_0067_20epoc/0.0067checkpoint_best.pth.tar\
    -q 6 \
    -i sun360test_500 \
    -o recon_test500_0067_20epoc \
    --result benchmark_test500_0067_20epoc_train30.json \
    --cuda \
    --real \
    --verbose



////////////////// treino atual: com w-mse + crop width 

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 python train.py \
    -d sun30 \
    --lambda 0.0067 \
    --checkpoint checkpoints360/check_sun30/check_0067_20epoc/0.0067checkpoint_best.pth.tar \
    --epochs 23 \
    --learning-rate 0.00001 \
    --lr_epoch 100 \
    --batch-size 4 \
    --patch-size 256 256 \
    --num-workers 12 \
    --cuda \
    --save_path checkpoints360/check_sun30/check_w-mse_cropRandom \
    --save \
    2>&1 | tee checkpoints360/check_sun30/check_w-mse_cropRandom/train360_mse.log


PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 python -u train360.py \
    -d sun30 \
    --lambda 0.0067 \
    --checkpoint checkpoints360/check_sun30/check_0067_20epoc/0.0067checkpoint_best.pth.tar \
    --epochs 23 \
    --learning-rate 0.00001 \
    --lr_epoch 100 \
    --batch-size 4 \
    --patch-size 256 256 \
    --num-workers 12 \
    --cuda \
    --save_path checkpoints360/check_sun30/check_w-mse_cropRandom \
    --save \
    2>&1 | tee checkpoints360/check_sun30/check_w-mse_cropRandom/train360_wmse.log


//pra ver a época do latest
    python - <<'PY'
import torch

path = "0.0067checkpoint_best.pth.tar"

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

