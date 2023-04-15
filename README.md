# CVComp - Compares Computer Vision Models

This project is a proof of concept for comparing different computer vision models and their performance.

It supports running various models locally using multiprocess and remotely using redis pubsub.

## Before setting up the environment
Note that you may need to install a rust compiler in order to build packages for particular platforms.
Visit https://rustup.rs for more information.

## Manually setting up the environment using Anaconda
Create a new Python environment based on Python 3.10 using Anaconda and activate it.

*Note*: Environment name is `cvcomp`
```shell
conda install -n cvcomp "numpy<1.24"
python -m pip install "opencv-python<4.8" 
python -m pip install redis --upgrade-strategy only-if-needed
```

### Tensorflow
Please install an adequate distribution of tensorflow for your platform or follow the instructions below
if available for your platform.

#### MacOS ARM (M1) - Apple Silicon
Guides:
- [TensorFlow](https://developer.apple.com/metal/tensorflow-plugin/) 
- [PyTorch](https://developer.apple.com/metal/pytorch/)
```shell
python -m pip install tensorflow-macos tensorflow-metal --upgrade-strategy only-if-needed
```

### Finalizing environment setup
```shell
python -m pip install tensorflow-hub --upgrade-strategy only-if-needed
```

## Automatically setting up environment using Anaconda
### On Apple Silicon (M1)
```shell
conda env create -f environment-m1.yml
```

## Downloading YoloV4 data

- [yolov4.cfg](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.cfg)
- [yolov4.weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights)
- [coco.names](https://github.com/pjreddie/darknet/blob/master/data/coco.names)

Add the files to the `data` folder with the following structure

```
project_root
├── data
│   ├── coco.names
│   ├── cfg
│   │   └── yolov4.cfg
│   └── weights
│       └── yolov4.weights    
```

## Running the application

Before running the demos, please visit the config.py file if you want to change the default settings

The application contains 2 runnable demo scripts

### run_locally.py

This will run the models locally on your machine using multiprocessing

```shell
PYTHONPATH=$PYTHONPATH:./src python ./src/run_locally.py
```

### run_redis.py

This will run the models on a remote machine using redis pubsub

Before running the script, please make sure you have a redis server running on the default port 6379

Using redis-cli run the following command to increase the pubsub limits as follows hard 256mb, soft 128mb

```shell
redis-cli CONFIG SET client-output-buffer-limit "normal 0 0 0 slave 268435456 67108864 60 pubsub 268435456 134217728 60"
```

You may now run the distributed demo

```shell
PYTHONPATH=$PYTHONPATH:./src python ./src/run_redis.py
```

# TODO
- [ ] refactor and clean up code
- [ ] handle startup and shutdown for the distributed components
- [ ] change from sending binary data to sending just bounding boxes and confidence scores when using redis
- [ ] add more models
- [ ] add more demos
- [ ] create docker images
- [ ] add GitHub workflows for building images
