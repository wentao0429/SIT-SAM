python train_cls.py \
 --task_name '117_turbo' \
 --checkpoint /path/to/checkpoint \
 --num_workers 64 \
 --num_epochs 20 \
 --batch_size 16 \
 --lr 3e-4 \
 --num_classes 117 \
 --accumulation_steps 4 \
 --gpu_ids 0 1 2 3 \
 --multi_gpu \
 --port 29500 \
 --resume \
 --KNN

