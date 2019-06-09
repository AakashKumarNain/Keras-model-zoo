# Keras Model Zoo

This repository contains pre-trained models that are readily available to use in `tf.keras` or `Keras`. 
There are two things that we are focusing on right now: `Computer Vision` and `NLP`, each of which is a `sub-package`.

# Objective and Motivation

Deep learning has really taken off in the last few years. Since `AlexNet` won the 2012 ImageNet competition, the amount of research and the papers/neural network architectures coming out has seen an exponential rise. It is also well known that `transfer learning` and `fine-tuning` both have been a great success and are used on a daily basis by Researchers and ML practitioners. In order to experiment fast, people rely on ready-made models. For example, if someone is trying to make some changes to the `Transformer` architecture, that person wants a ready-made model with a certain amount of flexibility. This not only helps saving time but it also encourages people to try more and more things. 

# Existing work

There have been many efforts made in this direction. For example,
* keras-applications
* slim-model-zoo
* tensor2tensor
* TFHub

# Why a new keras_model_zoo then?

All of the existing works listed above have certain limitations.
* `keras-applications` is limited to a very small number of models and doesn't provide pre-trained models for `NLP`, or `Object Detection`, or `Semantic Segmentation`. 
* `TFHub` is even more limiting. It doesn't provide the flexibility researchers need. As of now, it is well suited for `inferencing` or little changes but not much. 
* `slim` isn't the recommended API in `TF2.0`
* `tensor2tensor` has its own learning curve and again, it doesn't provide all the functionalities

# Requirements
* The models should be compatible with both, `tf.keras` and `Keras`.
* Models should be described as Lego blocks that are usable and which can be modified easily. 
* Models focusing on computer vision should go under `KerasCV` and others, focusing on NLP, should go under `KerasNLP`.
* Please refer to the `API signatures` defined in each sub-package to for more details
