# Head-Free Lightweight Semantic Segmentation with Linear Transformer

This repository contains the official Pytorch implementation of training & evaluation code and the pretrained models for [AFFormer](https://arxiv.org/abs/xxxx).🔥🔥

<!-- ![image](docs/figure1.png) -->

<div align="center">
  <img src="./docs/figure1.png" height="400">
</div>
<p align="center">
  Figure 1: Performance of AFFormer.
</p>

AFFormer is a head-free, lightweight and powerful semantic segmentation method, as shown in Figure 1.

We use [MMSegmentation v0.21.1](https://github.com/open-mmlab/mmsegmentation/tree/v0.21.1) as the codebase.



## Installation

For install and data preparation, please refer to the guidelines in [MMSegmentation v0.21.1](https://github.com/open-mmlab/mmsegmentation/tree/v0.21.1).

An example (works for me): ```CUDA 11.3``` and  ```pytorch 1.10.1```

```
pip install mmcv-full==1.5.0
pip install torchvision
pip install timm
pip install opencv-python
pip install einops
```

## Evaluation

Download `weights`
(
[google drive](https://drive.google.com/drive/folders/1Mru24qPdta9o8aLn1RwT8EapiQCih1Sw?usp=share_link) |
[alidrive](https://www.aliyundrive.com/s/Ha2xMsG9ufy)
)

Example: evaluate ```AFFormer-base``` on ```ADE20K``` :

```
# Single-gpu testing
bash tools/dist_test.sh ./configs/AFFormer/AFFormer_base_ade20k.py /path/to/checkpoint_file.pth 1 --eval mIoU

# Multi-gpu testing
bash tools/dist_test.sh ./configs/AFFormer/AFFormer_base_ade20k.py /path/to/checkpoint_file.pth <GPU_NUM> --eval mIoU

# Multi-gpu, multi-scale testing
bash tools/dist_test.sh ./configs/AFFormer/AFFormer_base_ade20k.py /path/to/checkpoint_file.pth <GPU_NUM> --eval mIoU --aug-test
```

## Training

Download `weights`
(
[google drive](https://drive.google.com/drive/folders/1Mru24qPdta9o8aLn1RwT8EapiQCih1Sw?usp=share_link) |
[alidrive](https://www.aliyundrive.com/s/Ha2xMsG9ufy)
)
pretrained on ImageNet-1K, and put them in a folder ```pretrained/```.

Example: train ```AFFormer-base``` on ```ADE20K```:

```
# Single-gpu training
bash tools/dist_train.sh ./configs/AFFormer/AFFormer_base_ade20k.py

# Multi-gpu training
bash tools/dist_train.sh ./configs/AFFormer/AFFormer_base_ade20k.py <GPU_NUM>
```

## Visualize

Here is a demo script to test a single image. More details refer to [MMSegmentation's Doc](https://mmsegmentation.readthedocs.io/en/latest/get_started.html).

```shell
python demo/image_demo.py ${IMAGE_FILE} ${CONFIG_FILE} ${CHECKPOINT_FILE} [--device ${DEVICE_NAME}] [--palette-thr ${PALETTE}]
```

Example: visualize ```SegFormer-B1``` on ```CityScapes```:

```shell
python demo/image_demo.py demo/demo.png local_configs/segformer/B1/segformer.b1.512x512.ade.160k.py \
/path/to/checkpoint_file --device cuda:0 --palette cityscapes
```

## License

The code is released under the MIT license.

## Copyright

Copyright (C) 2010-2021 Alibaba Group Holding Limited.

## Citation

If you find this work helpful to your research, please consider citing the paper:

```bibtex
@inproceedings{dong2023afformer,
  title={AFFormer: Head-Free Lightweight Semantic Segmentation with Linear Transformer},
  author={Bo, Dong and Pichao, Wang and Fan Wang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  pages={},
  year={2023}
}
```

### Code will be available before 12.30!
