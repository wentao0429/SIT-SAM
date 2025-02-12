python train_softmax.py \
 --task_name 'train_softmax_test_60' \
 --checkpoint /path/to/checkpoint \
 --num_workers 64 \
 --num_epochs 20 \
 --batch_size 2 \
 --lr 3e-4 \
 --accumulation_steps 16 \
  --port 13121 \
 --group_start 60 \
 --gpu_ids 0 1 2 3 \
 --multi_gpu \
# --resume




