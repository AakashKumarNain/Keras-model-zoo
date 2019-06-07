# Design Proposal

## Interface

## Classification Models
`Classification` models serve as the `hello world` of Computer Vision with Deep Learning. The models defined here should follow the same norm as the ones defined in `keras-applications`. The same API can be extended for some level of flexibility though. This is a list(not exhaustive) of all the classification models, excluding that are already in `keras-applications` as they can be directly ported here with minimal changes. 
* Inception V1
* Inception V2
* Inception V4
* VGG16_with_BN
* VGG19_with_BN
* Mobilenet V3
* ResNet-18
* ResNet-34
* M-NasNet
* ShuffleNet
* SqueezeNet
* WideResNet
* P-NasNet

#### Example
```python
from keras_model_zoo.vision.classification import mobilenet_v3

# load the pre-trained model
model = mobilenet_v3.MobileNetV3(weights='imagenet')

# compile and fit
model.compile(..)
model.fit(...)    # or model.fit_generator(...)

```
There should be some notable changes though. 
1. In the pretrained models, layers that have `multiple modes` should be defined clearly keeping in mind that these models are going to be used for `transfer learning` and `fine-tuning`. So, the `training mode` and `inference mode` when `freezing` a layer should be something very clear.
2. Sometimes we don't want the weights to be loaded for all the layers. For example, in VGG16, instaed of loading weights for all the conv blocks, what if we want to load weights only for the first four conv blocks 
