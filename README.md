# Realtime_Object_Detection
Codebase for realtime object detection

**News**: With the help of this codebase, we win the second place in the [2019 DAC System Design Contest](http://www.cse.cuhk.edu.hk/~byu/2019-DAC-SDC/index.html).

## Introduction
This codebase is modified based on [mmdetection](https://github.com/open-mmlab/mmdetection) 
and the difference between them is that some two stage object detection algorithms are removed 
and some small and compact backbones are added.

The purpose of this codebase is to facilitate network quantization and pruning, 
eventually applying deployment on different hardware devices.

## Installation

### Requirements

- Linux
- Python 3.5+ 
- CUDA 9.0+
- NCCL 2+
- GCC 4.9+

### Step by step installation

a. Create a conda virtual environment and activate it. 
```shell
conda create -n realtime_object_detection python=3.7 -y
source activate realtime_object_detection
```
b. Install some requirements.
```shell
pip install torch torchvision
pip install cython
```
c. Clone the realtime_object_detection repository.

```shell
git clone https://github.com/waterbearbee/realtime_object_detection
cd realtime_object_detection
```

d. Compile cuda extensions.

```shell
sh ./compile.sh
```

e. Install other dependencies.

```shell
python3 setup.py develop
```

## Getting Started

### Training

#### Train with a single GPU

```shell
python3 ./tools/train.py ${CONFIG_FILE} 
```

#### Train with multiple GPUs

```shell
python3 -m torch.distributed.launch --nproc_per_node=${GPU_NUM} ./tools/train.py ${CONFIG_FILE} --launcher pytorch
```

### Testing

#### Test with a single GPU
```shell
python3 ./tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] --eval bbox 
```

#### Test with multiple GPUs
```shell
python3 -m torch.distributed.launch --nproc_per_node=${GPU_NUM} ./tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --out ${RESULT_FILE} --eval bbox --launcher pytorch
```

## DAC Competition Results (RetinaNet)

|     Backbone       |   Neck  |   Head     |  box AP  |
| :-----------------:| :-----: | :---------:| :------: | 
|  MobileNetv2_x1_0  |  P3-P5  | 2conv(64c) |   84.2   |   
|  MobileNetv2_x0_5  |  P3-P5  | 2conv(64c) |   79.9   |
|  MobileNetv2_x0_25 |  P3-P5  | 2conv(64c) |   74.4   |
|  ShuffleNetV2_x1_0 |  P3-P5  | 2conv(64c) |   80.6   |
|  ShuffleNetV2_x0_5 |  P3-P5  | 2conv(64c) |   75.0   |
