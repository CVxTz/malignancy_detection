# Metastasis Detection Using CNNs, Transfer Learning and Data Augmentation

The objective of this project is to detect Cancer metastasis on histopathology
images of lymph nodes using the PatchCamelyon dataset
[[1]](https://github.com/basveeling/pcam) hosted on Kaggle.

Having the correct diagnosis of the advancement of the disease is crucial to
choose the most suitable treatment course, this is why doctors rely on
histopathology images of biopsied tissue where there might be metastasis. In
this project we will train a model to automatically detect evidence of
malignancy in order to help doctors make better decisions and hopefully provide
better care for Cancer patients.

The steps presented here can also serve a strong baseline to any generic image
classification problem :

### Data

The dataset is a set of 96x96 images where each image is labeled 1 if there is
evidence of malignancy in the 32x32 center section of the image or 0 otherwise.

![](https://cdn-images-1.medium.com/max/1200/1*994tgmGJLssN4R4076QEog.png)

<span class="figcaption_hack">Examples from the training set</span>

### Augmentation

In order to reduce overfitting and increase the generalization capabilities of
the model we use data augmentation, which is a sequence of random perturbations
applied to the image that preserve the label information. Training with those
perturbation also makes the model more robust to noise and increase its
invariance to translation and rotation.

![](https://cdn-images-1.medium.com/max/1200/1*pYIVaZQPxLvK0RRg0psJSQ.png)

<span class="figcaption_hack">Random Augmentations applied to the same input image</span>

### Model

We use NasNet mobile pretrained on ImageNet (see [Transfer
Learning](https://en.wikipedia.org/wiki/Transfer_learning) ) because its fast
and thus can be fully trained on Kaggle kernels within the 6 hours time limit.

![](https://cdn-images-1.medium.com/max/1200/1*g-QOg_7Rpm0VZ1E2uYmo4g.png)

### Training

We use a small portion of the training set as validation, and then use model
[Checkpoint](https://keras.io/callbacks/#modelcheckpoint) Keras Callback to save
the best weights and load them before we do the prediction on the Leaderboard
data.

### Prediction and Post processing

For each image of the test set we average the predictions of the original image
and that of the horizontally/vertically flipped versions.

<span class="figcaption_hack">Leaderboard Result</span>

This approach achieves an
[AUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) score of
**0.9709** which is comparable to the state-of-the-art ([ Rotation Equivariant
CNNs for Digital Pathology](https://arxiv.org/abs/1806.03962)) of **0.963**.

You can run the model online on Kaggle :
[https://www.kaggle.com/CVxTz/cnn-starter-nasnet-mobile-0-9709-lb](https://www.kaggle.com/CVxTz/cnn-starter-nasnet-mobile-0-9709-lb)

Github Repo :
[https://github.com/CVxTz/malignancy_detection/tree/master/code](https://github.com/CVxTz/malignancy_detection/tree/master/code)

[1] [https://github.com/basveeling/pcam](https://github.com/basveeling/pcam)
