COMPILANDO NO DEEPCOOL

Lalic original (treinado com 40 épocas e 400 mil imagens planas) no dataset ctc

python eval.py \
    -m LALIC \
    -p checkpoints/lalic-q1.pth checkpoints/lalic-q3.pth checkpoints/lalic-q6.pth \
    -q 1 3 6 \
    -i CTC-360-resized \
    -o recon_ctc/ctc_lalic_original \
    --result benchmark_ctc/benchmark_3q_original.json \
    --cuda \
    --real \
    --verbose