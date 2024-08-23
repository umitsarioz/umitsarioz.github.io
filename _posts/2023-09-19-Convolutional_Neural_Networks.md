---
date: 2023-09-19
title: Convolutional Neural Networks
image: /assets/img/post4/conv_1.jpeg
categories: [Machine Learning,Supervised Learning]
tags: [Machine Learning,Algorithms, Neural Networks]
---

Convolutional Neural Networks (CNNs) are a type of artificial neural network designed to perform feature extraction from image data. The term "convolution" comes from the mathematical operation applied between matrices. When humans encounter an object, they distinguish it by relying on previously learned features. Since neural network structures are inspired by the human brain, CNNs have been developed to emulate this capability successfully. The CNN architecture is composed of several layers:

## 1. Convolution Layer

This is the layer where the convolution operation occurs. The convolution operation is essentially a form of matrix multiplication. The image input is processed by a matrix known as a kernel (or filter), generating new results. The operation is then repeated across all pixel values by shifting according to a value known as the stride. These results represent the features extracted from the input image. The features extracted vary depending on the kernel (filter) matrix used.
<br><b>Stride:</b> The number of steps the filter moves over the input matrix. By default, the stride value is 1.

![Convolution_Operation_1](/assets/img/post4/conv_operation_1.gif)
_Example of a convolution operation with a stride value of 1_

To further explain the concept of image input: every image has specific pixel values, which are numerical values within a matrix. These pixels form the various elements of the image. Colored images are composed of three channels: Red (R), Green (G), and Blue (B), commonly referred to as RGB. In cases where there are three channels, the convolution operation is carried out as shown in the example below.

![Convoluiton_Operation_2](/assets/img/post4/conv_operation_2.gif)
_Three Channel Convolutional Operation with 1 Stride_

## 2. Pooling Layer

![Pooling_Layer](/assets/img/post4/conv_pooling_2.jpeg)
_Pooling Types_ 

In this layer, a pooling matrix is applied to the input matrix. This matrix does not have any weights or bias values, but it reduces the number of parameters in the model, which increases the model's efficiency. Pooling is typically used with the padding value set to "same," while the default value is "valid."

![Padding_Sample](/assets/img/post4/conv_padding_1.png)
_Example of padding with a value of 0 and a size of (2x2)_

<br><b>Valid padding:</b> No extra values are added to the input; it is used as is.
<br><b>Same padding:</b> A 2x2 border is added outside the input, preventing any data loss.
<br><b>Padding value calculation:</b> $$ (f-1) / 2 $$ 
> f: filter (kernel) matrix size

<br>

<b>Output Size Calculation:</b> $$\left\lfloor \frac{n + 2p - f}{s} \right\rfloor + 1 $$ 
> n: input size <br>
> p: padding <br>
> f: filter (kernel) size <br>
> s: stride value <br>

## 3. Normalization Layer

In this layer, normalization is applied to the input. This process compresses the distribution of data to enhance its readability. Typically, normalization in an RGB image involves dividing by 255 since there are 256 pixel values.

## 4. Fully-Connected Layer

Within this layer, the matrices containing input data are converted into vectors.

### 4.1 Flatten Layer

Before being sent to the fully connected layer, the data is flattened into a single vector. This step vectorizes the inputs.

![Flatten_Layer](/assets/img/post4/conv_flatten_1.png)
_Flatten Layer_ 

### 4.2 Dense Layer
This is where the fully connected layer structure is established. It can be considered as the point where the model returns to its standard form.

There are various CNN architectures, with some of the most commonly used being:
- LeNet
- AlexNet
- VGGNet
- GoogLeNet
- ResNet