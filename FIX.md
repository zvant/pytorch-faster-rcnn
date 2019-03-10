## Major Modification

Fixed some bugs caused by older version of ROI_Align, make sure to compile and install it from GitHub. Tested on commit ``d85fa43dacc29704c1b93de52deef006ed7530ac``.

## Environment

Ubuntu 14.04.5 LTS
Python 2.7.15
CUDA 10.0

## HOWTO:

1. Download [PyTorch](https://github.com/pytorch/pytorch).
`$ git clone --recursive --branch v0.4.0 https://github.com/pytorch/pytorch.git`Compile and install it according to instructions.

2. Download [RoIAlign.pytorch](https://github.com/longcw/RoIAlign.pytorch). Checkout to tested commit, then compile and install.

3. Compile pytorch-faster-rcnn.


