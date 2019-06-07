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
2. Sometimes we don't want the weights to be loaded for all the layers. For example, in VGG16, instaed of loading weights for all the conv blocks, what if we want to load weights only for the first conv block only 

```python
# This is actually a breaking change to the existing API.
# So we need to compe up with some alternative. One way to include
# this is to provide a function in the `utils` file if possible
model.load_weights("weights.h5", by_name=True, weights_for_layers=['Conv1_1', 'Conv1_2'])
```
 
---

## Detection Models

* SSD
* SSDLite
* RetinaNet
* Yolo v1/v2/v3
* FRCNN
* RFCN

Any detection network comprises of the following parts
* A base network for feature extraction
* Auxillary convolutions
* Predictor layers
* An anchor-generator layer
* Box encoders/decoders
* Non-Maximum Supression
* Utilities like `Hard-Negative mining`, `Jaccard Overlap`

 Consider these as the lego blocks for building a detection model. Apart from `feature extraction`, most of the moving parts remain the same for a particular type of model. For example, there can be `MobileNet-SSD` as well as `Inceptionv3-SSD`. Only the base network changes, rest of the pipleine for a `SSD` model remains the same.

The benefit of this is that we can use a lot of functionality from the `Classification` subpackage, until unless the base network is something that is not readily available in `Classification` subpackage. It is also necessary to make sure that changing things in other parts is easy as well. For example, if we want to change the `scales` and `aspect_ratios` of the anchors, then we can these values in the function/layer responsible for creating anchors

### Example
```python
from keras_model_zoo.vision.detection import SSD
from keras_model_zoo.vision.detection.utils import generate_anchors
from keras_model_zoo.vision.detection.utils import box_encoder, box_decoder, nms

# define your base_model 
base_model = Model(...)

# add auxillary conv layers if needed
base_model_output = base_model.output
x = Conv2D(...)(base_model_output)
x = Conv2D(...)(x)
...

# add predictor layers for boxes
loc1 = Conv2D(n_boxes_loc1*4, (3,3), padding="same")(conv4_3)
loc2 = Conv2D(n_boxes_loc2*4, (3,3), padding="same")(conv7)
....

# add predictor layers for classes
classes_conv4 = Conv2D(n_boxes_loc1*nb_classes, (3,3), padding="same")(conv4_3)
classes_conv7 = Conv2D(n_boxes_loc2*nb_classes, (3,3), padding="same")(conv7)
...

# generate anchors. The anchors specification should be defined
# in a dict for each of the layer for which priors have to be generated
anchors_conv4 = generate_anchors(config={name:'conv4', 
                                         fmap_dim:(8,8), 
                                         scale:0.2,
                                         ar=[1., 2., 3., 1/2., 1/3.]
                                         extra_prior:True})
...
                                         
                                         

# define ssd model now
model = SSD(base_model=base_model,
            loc_layers=[loc1, loc2],
            class_pred_layers=[classes_conv4, classes_conv7],
            anchors=[anchors_conv4, anchors_conv7],
            enoder=box_endoer #can be user defined as well,
            decoder=box_decoder #can be user defined as well,
            hard_negative_mining=True,
            nms_type="all" #can be class wise as well)

# train model
model.fit(..)
```
