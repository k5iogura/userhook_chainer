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

Check,  
```
 $ python3
 >>> import chainer as chainer
 >>> chainer.backends.cuda.available
 True
 >>> chainer.backends.cuda.cudnn_available
 True
```

