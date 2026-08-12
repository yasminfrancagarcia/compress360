Comando para compilar o compressor 

python eval.py \
    -m LALIC \
    -p checkpoints/lalic-q1.pth checkpoints/lalic-q2.pth checkpoints/lalic-q3.pth checkpoints/lalic-q4.pth checkpoints/lalic-q5.pth checkpoints/lalic-q6.pth \
    -q 1 2 3 4 5 6 \
    -i dataset \
    -o recon_images \
    --result benchmark_teste5.json \
    --cuda \
    --real \
    --verbose


treinar 
CUDA_VISIBLE_DEVICES=0 python train.py \
    -d /path/to/dataset \
    --lambda 0.0067 \
    --epochs 40 \
    --lr_epoch 36 \
    --batch-size 8 \
    --save_path /path/for/saving --save


for lmbda in 0.0018 0.0067 0.0130 0.0250 0.0483; do
    CUDA_VISIBLE_DEVICES=0 python train.py \
        -d /path/to/dataset \
        --lambda $lmbda \
        --epochs 40 \
        --lr_epoch 36 \
        --batch-size 8 \
        --save_path /path/for/saving --save
done