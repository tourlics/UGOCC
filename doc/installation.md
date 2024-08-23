# installation

Following https://mmdetection3d.readthedocs.io/en/dev/compatibility.html



**a. Create a conda virtual environment and activate it.**
```shell
conda create -n UGOCC python=3.8 -y
conda activate UGOCC
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**
```shell
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu114  -f https://download.pytorch.org/whl/torch_stable.html
# Recommended torch>=1.12
```

**c. Install mmcv-full.**
```shell
pip install mmcv-full==1.5.2

```

**d. Install mmdet and mmseg.**
```shell
pip install mmdet==2.24.0
pip install mmsegmentation==0.24.0
```

**e. Install mmdetection from our source code.**
```shell
cd mmdetection3d
pip install -v -e .
# python setup.py install
```


**h. Prepare pretrained models.**
```shell
mkdir checkpioints
cd  checkpioints
download resnet50-0676ba61.pth from pytorch offical website
```

**h. compile plugins**
```shell
cd projects
python setup.py install
```