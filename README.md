usage:
python -m torch.distributed.launch --nproc_per_node=2 main.py \
  --pretrained \
  --freeze-backbone \
  --progressive-unfreeze \
  --unfreeze-epoch 50 \
  --data-dir path/to/food-101 \
  --output-dir ./output \
  --batch-size 32 \
  --epochs 100
