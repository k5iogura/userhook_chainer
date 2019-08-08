# Modify process of cnn via user hook with chainer  

[how to visualize filters and featuremap in cnn](https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks/)  

### To use CUDA GPU and cuDNN library

With prerequisites below,  
- CUDA  
- cuDNN  
CUDA and cuDNN Versions must be matched such as 9.0.  

Prepare chainer and cupy,  
```
 # python3 -m pip install setuptools
 # python3 -m pip install chainer

 # python3 -m pip install cupy-cuda90 # if your CUDA version is 9.0
```

Setup user environmental variables in .bashrc for CUDA,  
```
export LD_LIBRARY_PATH=/usr/local/lib:./:$LD_LIBRARY_PATH
export CUDA_PATH=/usr/local/cuda
export CUDA_ROOT_DIR=/usr/local/cuda
export PATH=$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
```

Check,  
```
 $ python3
 >>> import chainer as chainer
 >>> chainer.backends.cuda.available
 True
 >>> chainer.backends.cuda.cudnn_available
 True
```

