# RD-Net
This repository is the official implementation of RD-Net.

## Prepare the dataset 
dataset_root
├─ kvasir_SEG
│  ├── Train
│  │   ├── images
│  │   └── masks 
│  └── Test
│      ├── images
│      └── masks
├─ CVC_ClinicDB
│  ├── Train
│  │   ├── images
│  │   └── masks 
│  └── Test
│      ├── images
│      └── masks
├─ ...

## Training
python train.py

## Testing
python eval.py


