# RD-Net
This repository is the official implementation of RD-Net.

## Prepare the dataset 
The dataset should be organized as follows:
The depth can be obtained directly using the [Depth Anything model](https://github.com/LiheYoung/Depth-Anything).
```
root/
├── kvasir_SEG/
│   ├── Train/
│   │   ├── images/
│   │   ├── masks/
│   │   ├── depth/         
│   │   └── depth_rgb/      
│   └── Test/
│       ├── images/
│       └── masks/
├── CVC_ClinicDB/
│   ├── Train/
│   │   ├── images/
│   │   ├── masks/
│   │   ├── depth/         
│   │   └── depth_rgb/    
│   └── Test/
│       ├── images/
│       └── masks/
└── ... 
```

## Training

python train.py

## Testing
python eval.py


