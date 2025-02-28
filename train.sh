CUDA_VISIBLE_DEVICES=4,5,6,7 \
python train.py --dist-url 'tcp://localhost:12754' \
    --dataset spaq csiq livec koniq10k kadid live \ 
    --multiprocessing-distributed --world-size 1 --rank 0 \
    --batch_size 44 --epochs 50 --seed 2024 \
    --random_flipping_rate 0.1 --random_scale_rate 0.5 \
    --model promptiqa \
    --save_path ./Exp/PromptIQA_2024
