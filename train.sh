python train.py \
 --task_name 'from_turbo117' \
 --checkpoint /path/to/checkpoint \
 --num_workers 80 \
 --num_epochs 30 \
 --batch_size 5 \
 --lr 3e-5 \
 --accumulation_steps 96 \
 --gpu_ids 0 1 2 3 \
 --multi_gpu \
 --resume




