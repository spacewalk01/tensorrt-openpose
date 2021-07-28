# TensorRT Implementation of OpenPose
This repo provides C++ implementation for OpenPose, human pose detection algorithm based on TensorRT framework that runs on Windows Machine. It is based on [pose detection](https://github.com/NVIDIA-AI-IOT/trt_pose) program developed by NVIDIA. Pose detection model runs at up to 500 FPS on RTX-3070 GPU with 224x224 ResNet input size. 

![example-gif-1](results/test1.gif)
![example-gif-2](results/test2.gif)

## Requirements
The following environment was set for the experiment but if you have a different Graphic Card, you need to download and install TensorRT / CUDA that matches your GPU version.
- Windows 10
- Visual Studio 2017
- RTX 3070 GPU
- TensorRT 7.2.1.6
- CUDA 11.1, Cudnn 8
- Python 3.7
- Torch 1.8.1
- OpenCV 4.5.1 with CUDA

## Installation
- Install PyTorch 
```
pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```
- Download and install [CUDA](https://developer.nvidia.com/cuda-11.1.0-download-archive) and [CUDNN](https://developer.nvidia.com/cudnn) by following this [installation guide](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html).
- Install [OpenCV 4.5.1 or 4.5.0]
- Download the [TensorRT](https://developer.nvidia.com/nvidia-tensorrt-download) zip file that matches your Windows version.
- Install TensorRT by copying the DLL files from <tensorrt_path>/lib to your CUDA installation directory. For more information, refer to [TensorRT installation guideline](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html).
```
move <TensorRT_Installpath>/lib/*.dll "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.1/bin"
```
- Install [trt_pose](https://github.com/NVIDIA-AI-IOT/trt_pose) for parsing the trained pytorch model to an onnx graph model. For more information, refer to [trt_pose installation guide](https://github.com/haotian-liu/yolact_edge/blob/master/INSTALL.md).
```
git clone https://github.com/NVIDIA-AI-IOT/trt_pose
cd trt_pose
python setup.py install
```
- Clone this github repository 
```
cd ..
git clone https://github.com/batselem/Human_pose_detection
cd Human_pose_detection
```
- Download a pretrained resnet model, [resnet18_baseline_att_224x224_A](https://drive.google.com/file/d/1XYDdCUdiF2xxx4rznmLb62SdOUZuoNbd/view) and put it in the current project folder.
- Create an onnx model, optimize the model using TensorRT and build a runtime inference engine.
```
python convert2onnx.py -i resnet18_baseline_att_224x224_A_epoch_249.pth -o trt_pose.onnx
<tensorrt_path>/bin/trtexec.exe --onnx=trt_pose.onnx --explicitBatch --saveEngine=trt_pose_fp16.engine --fp16
```
- Open the solution with Visual Studio. Select `x64` and `Release` for the configuration and start building the project. 
## Training
For training a larger model, you may refer to [link](https://docs.nvidia.com/isaac/isaac/packages/skeleton_pose_estimation/doc/2Dskeleton_pose_estimation.html)
## References
  - https://github.com/NVIDIA-AI-IOT/trt_pose - Real-time pose estimation (Python and C++)
  - https://github.com/CaoWGG/TensorRT-YOLOv4 - Object detection based on Tensorrt (C++)

