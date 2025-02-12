python train_cls.py \
 --task_name 'test_ef' \
 --checkpoint /path/to/checkpoint \
 --num_workers 80 \
 --num_epochs 10 \
 --batch_size 1 \
 --lr 3e-4 \
 --num_classes 117 \
 --accumulation_steps 20 \
 --gpu_ids 0 1 2 3 \
 --multi_gpu \
# --resume
