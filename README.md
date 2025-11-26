# RD-Net
This repository is the official implementation of RD-Net.

## Training

python train.py \
  --root ./dataset \
  --dataset kvasir_SEG \
  --labeled_perc 10 \
  --total_iter 40000 \
  --gpu_id 0

## Testing

python eval.py


We first release the core code for the methods proposed in the paper to facilitate comparison.
The remaining code and implementation details are in the process of being finalized and will be released soon.
