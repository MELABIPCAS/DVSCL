## Micro-Expression Recognition using Dual-View Self-Supervised Contrastive Learning with Intensity Perception

This repo covers the implementation for Micro-Expression Recognition using Dual-View Self-Supervised Contrastive Learning with Intensity Perception (DVSCL):

software developer： [Haoliang Zhou](https://github.com/HaoliangZhou)


## Installation

This repo was tested with Ubuntu 16.04.5 LTS, Python 3.5, PyTorch 0.4.0, and CUDA 9.0. But it should be runnable with recent PyTorch versions >=0.4.0

**Note:** It seems to us that training with Pytorch version >= 1.0 yields slightly worse results. If you find the similar discrepancy and figure out the problem, please report this since we are trying to fix it as well.

## Data preparation
We used the [CAS(ME)<sup>3</sup>](https://ieeexplore.ieee.org/abstract/document/9774929) dataset for pre-training and fine-tuning, the data lists are reorganized as follow:

```
data/
├─ v_depth/
│  ├─ spNO1_test.txt
│  ├─ spNO1_train.txt
│  ├─ spNO2_test.txt
│  ├─ ...
│  ├─ spN217_train.txt
├─ v_rgb/
│  ├─ spNO1_test.txt
│  ├─ spNO1_train.txt
│  ├─ spNO2_test.txt
│  ├─ ...
│  ├─ spN217_train.txt
│  ├─ subName.txt
```
1. There are 3 columns in each txt file: 
```
casme3_diff_me4/casme3_diff_imgs_all/others/spNO.166-b-837-spNO.166-b-834_diff.png 1 0
```
In this example, the first column is the path of the rgb_diff image for a particular ME sample, the second column is the label (0-2 for three emotions), and the third column is the database type (0 for casme3 dataset).

2. There are 95 raws in _subName.txt_: 
```
spNO1
spNO10
spNO11
...
spNO9
```

## Pre-training AlexNet/ResNets with DVSCL on CASME3

**Note:** For AlexNet, we split across the channel dimension and use each half to encode RGB and Depth. For ResNets, we use a standard ResNet model to encode each view.

NCE flags:
- `--nce_k`: number of negatives to contrast for each positive. Default: 4096
- `--nce_m`: the momentum for dynamically updating the memory. Default: 0.5
- `--nce_t`: temperature that modulates the distribution. Default: 0.07 for ImageNet, 0.1 for STL-10

Path flags:
- `--data_folder`: specify the CASME3 data folder.
- `--model_path`: specify the path to save model.
- `--tb_path`: specify where to save tensorboard monitoring events.

Model flag:
- `--model`: specify which model to use, including *alexnet*, *resnets18*, *resnets50*, and *resnets101*

An example of command line for pre-training DVSCL (Default: `AlexNet` on Single GPU)
```
CUDA_VISIBLE_DEVICES=0 python pre-train_CVSCL.py 
 --batch_size 256 \
 --num_workers 36 \
 --data_folder /path/to/data \
 --model_path /path/to/save \
 --tb_path /path/to/tensorboard
```

Training DVSCL with ResNets use 4 GPUs, the command of using `resnet50v1` looks like
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python pre-train_CVSCL.py 
 --model resnet50v1 \
 --batch_size 128 \
 --num_workers 24 \
 --data_folder path/to/data \
 --model_path path/to/save \
 --tb_path path/to/tensorboard \
```

To support mixed precision training, simply append the flag `--amp`, which, however is likely to harm the downstream classification. I measure it on ImageNet100 subset and the gap is about 0.5-1%.


## Training Linear Classifier on CASME3

Path flags:
- `--data_folder`: specify the CASME3 data folder. Should be the same as above.
- `--save_path`: specify the path to save the linear classifier.
- `--tb_path`: specify where to save tensorboard events monitoring linear classifier training.

Model flag `--model` is similar as above and should be specified.

Specify the checkpoint that you want to evaluate with `--model_path` flag, this path should directly point to the `.pth` file.

An example of command line for evaluating, say `./models/alexnet.pth`, should look like:
```
CUDA_VISIBLE_DEVICES=0 python LinearProbing_CVSCL_10fold.py --dataset casme3 \
 --data_folder /path/to/data \
 --save_path /path/to/save \
 --tb_path /path/to/tensorboard \
 --model_path ./models/alexnet.pth \
 --model alexnet --learning_rate 0.1 \
 --layer 5
```

## Fine-tuning on CASME3

Path flags:
- `--data_folder`: specify the CASME3 data folder. Should be the same as above.
- `--save_path`: specify the path to save the linear classifier.
- `--tb_path`: specify where to save tensorboard events monitoring linear classifier training.

Model flag `--model` is similar as above and should be specified.

Specify the checkpoint that you want to evaluate with `--model_path` flag, this path should directly point to the `.pth` file.

An example of command line for evaluating, say `./models/alexnet.pth`, should look like:
```
CUDA_VISIBLE_DEVICES=0 python fine-tune_CVSCL.py 
 --dataset casme3 \
 --data_folder /path/to/data \
 --save_path /path/to/save \
 --tb_path /path/to/tensorboard \
 --model_path ./models/alexnet.pth \
 --model alexnet --learning_rate 0.1 \
 --layer 5
```


## Acknowledgements

Our code is based on [CMC](https://github.com/HobbitLong/CMC) repository. We thank the authors for releasing their code. If you use our model and code, please consider citing these works as well.
