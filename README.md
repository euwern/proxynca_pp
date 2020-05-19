
ProxyNCA++: Revisiting and Revitalizing Proxy Neighborhood Component Analysis
==============================================================================
This repo consists of the source code for the [ProxyNCA++ paper](https://arxiv.org/abs/2004.01113)

Make sure to download the corresponding dataset to the correct folder as specified in dataset/config.json
We also include script to convert the dataset to hdf5 format. .

To run the code
```
# CUB
CUDA_VISIBLE_DEVICES=0,1 python train.py --dataset cub  --config config/cub.json --mode train --apex --seed 0
CUDA_VISIBLE_DEVICES=0,1 python train.py --dataset cub  --config config/cub.json --mode trainval --apex --seed 0

# CARS
CUDA_VISIBLE_DEVICES=0,1 python train.py --dataset cars  --config config/cars.json --mode train --apex --seed 0
CUDA_VISIBLE_DEVICES=0,1 python train.py --dataset cars  --config config/cars.json --mode trainval --apex --seed 0

# SOP
CUDA_VISIBLE_DEVICES=0,1 python train.py --dataset sop  --config config/sop.json --mode train --apex --seed 0
CUDA_VISIBLE_DEVICES=0,1 python train.py --dataset sop  --config config/sop.json --mode trainval --apex --seed 0

# INSHOP
CUDA_VISIBLE_DEVICES=0,1 python train.py --dataset inshop  --config config/inshop.json --mode train --apex --seed 0
CUDA_VISIBLE_DEVICES=0,1 python train.py --dataset inshop  --config config/inshop.json --mode trainval --apex --seed 0

```

The following is the Bixtex of our paper:
```
@article{teh2020proxynca++,
  title={ProxyNCA++: Revisiting and Revitalizing Proxy Neighborhood Component Analysis},
    author={Teh, Eu Wern and DeVries, Terrance and Taylor, Graham W},
      journal={arXiv preprint arXiv:2004.01113},
        year={2020}
        }
```

