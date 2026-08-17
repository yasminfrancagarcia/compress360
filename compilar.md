
source .venv-lalic/bin/activate
Comando para compilar o compressor 

python eval.py \
    -m LALIC \
    -p checkpoints/lalic-q1.pth checkpoints/lalic-q2.pth checkpoints/lalic-q3.pth checkpoints/lalic-q4.pth checkpoints/lalic-q5.pth checkpoints/lalic-q6.pth \
    -q 1 2 3 4 5 6 \
    -i sun360test_500 \
    -o recon_imagesSun360_test \
    --result benchmark_testeSun500_test.json \
    --cuda \
    --real \
    --verbose


treinar 
CUDA_VISIBLE_DEVICES=0 python train360.py \
    -d dataset360 \
    --lambda 0.0025 \
    --epochs 40 \
    --lr_epoch 36 \
    --batch-size 4 \
    --save_path checkpoints360/ --save
    --patch-size 512 1024


for lmbda in 0.0018 0.0067 0.0130 0.0250 0.0483; do
    CUDA_VISIBLE_DEVICES=0 python train.py \
        -d /path/to/dataset \
        --lambda $lmbda \
        --epochs 40 \
        --lr_epoch 36 \
        --batch-size 8 \
        --save_path /path/for/saving --save
done

CUDA_VISIBLE_DEVICES=0 python train360.py \
    -d sun360 \
    --lambda 0.0025 \
    --epochs 40 \
    --lr_epoch 36 \
    --batch-size 4 \
    --patch-size 512 1024 \
    --save_path checkpoints360/ --save

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 python train360.py \
    -d sun360 \
    --lambda 0.0025 \
    --epochs 40 \
    --lr_epoch 36 \
    --batch-size 8 \
    --patch-size 512 1024 \
    --save_path checkpoints360/ --save

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 python train360.py \
    -d sun360/train1000 \
    --lambda 0.0025 \
    --epochs 5 \
    --lr_epoch 36 \
    --batch-size 1 \
    --patch-size 256 256 \
    --save_path checkpoints360/test10epoc/ --save

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


PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 python train360.py \
    -d sun360/train1000 \
    --lambda 0.00483 \
    --epochs 5 \
    --lr_epoch 36 \
    --batch-size 1 \
    --patch-size 256 256 \
    --save_path checkpoints360/ --save
    

python eval.py \
    -m LALIC \
    -p checkpoints/lalic-q1.pth checkpoints/lalic-q2.pth checkpoints/lalic-q3.pth checkpoints/lalic-q4.pth checkpoints/lalic-q5.pth checkpoints/lalic-q6.pth \
    -q 1 2 3 4 5 6 \
    -i sun360test_500 \
    -o recon_imagesSun360_test \
    --result benchmark_testeSun500_test.json \
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