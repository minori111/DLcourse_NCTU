# DLcourse_NCTU_Lab
# Lab 4: Various activations + Batch normalization + Weight Initialization

## A052653 黃翊軒

## Introduction
In this lab, I used ReLU, ELU activation functions, batch normalization, and weight initialization in NIN, and trained it on Cifar-10 dataset.

The main goal of this lab is to train a NIN using three different activation functions and compare
- 2(ReLU, ELU) X 2(w/wo BN) X 2(w/wo weight initial)
-  [ReLU, without BN, with initialization] and [ELU, without BN, with initialization] **do not converge**
 
The best accuracy is **91.36%** of the combination **[ELU, BN, without initialization]**
## Experiment setup
- Training Hyperparameters:
    - Method: **SGD** with **Nesterov** momentum
    - Mini-batch size: **128** (391 iterations for each epoch)
    - Total epochs: **164**, momentum **0.75** 
    - Initial learning rate: **0.1**, divide by 10 at 81, 122 epoch
    - Loss function: **cross-entropy**

- Data augmentation parameters:
    - Translation: Pad **4** zeros in each side and random cropping back to 32x32 size
    - Horizontal flipping: With probability **0.5**
- Data preprocessing
    - Normalize each color channel (compute from entire CIFAR10 training set)
```python
#Normalize each color channel
x_train[:,:,:,0] = (x_train[:,:,:,0]-125.3)/63.0
x_train[:,:,:,1] = (x_train[:,:,:,1]-123.3)/62.1
x_train[:,:,:,2] = (x_train[:,:,:,2]-113.9)/66.7
#x_test /= 255
x_test[:,:,:,0] = (x_test[:,:,:,0]-125.3)/63.0
x_test[:,:,:,1] = (x_test[:,:,:,1]-123.3)/62.1
x_test[:,:,:,2] = (x_test[:,:,:,2]-113.9)/66.7
```
- Weight initialization
    - kernel_initializer=he_normal(seed=None)
        - He normal initializer.
        - It draws samples from a truncated normal distribution centered on 0 with stddev = sqrt(2 / fan_in) where fan_in is the number of input units in the weight tensor.
    - First conv layer: Random_normal(stddev=0.01)
    - Others: Random_normal(stddev=0.05)
- ELU
    - ELU(alpha=1.0)
- Batch Normalization
    - BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)



## Result
First I present the best result and show the comparison of the six convergent combination : 
- ELU_noWI_noBN **(testing accuracy over 90%)**
- ELU_noWI_BN **(testing accuracy over 91%)**
- ELU_WI_BN **(testing accuracy over 90%)**
- Relu_noWI_BN **(testing accuracy over 90%)**
- Relu_noWI_noBN 
- Relu_WI_BN. **(testing accuracy over 90%)**

The details of each combinaiton performence will be shown after above.

### Best result [ELU, BN, without initialization]
The train loss is **0.1449**
The train accuracy is **95.04%**
The test loss is **0.2864**
The test accuracy is **91.36%**
The run time on GTX1060-6GB is **2:51:53** (h/m/s)
![](https://i.imgur.com/lhOlGGh.png)


### The comparison of the six convergent combination
- test accuracy
![](https://i.imgur.com/fLZ67fA.png)
- train accuracy
![](https://i.imgur.com/RGMmiVq.png)
- test loss
![](https://i.imgur.com/91MDtzD.png)
- train loss
![](https://i.imgur.com/MiOM7qp.png)

### The details of each combinaiton performence
#### ELU_noWI_noBN
The train loss is **0.1825**
The train accuracy is **93.52%**
The test loss is **0.3455**
The test accuracy is **90.69%**
The run time on GTX1060-6GB is **1:54:33** (h/m/s)
![](https://i.imgur.com/6K91J4l.png)
![](https://i.imgur.com/KSxDzHE.png)
![](https://i.imgur.com/g4DMTOx.png)

#### ELU_noWI_BN
The train loss is **0.1449**
The train accuracy is **95.04%**
The test loss is **0.2864**
The test accuracy is **91.36%**
The run time on GTX1060-6GB is **2:51:53** (h/m/s)
![](https://i.imgur.com/1P700ea.png)
![](https://i.imgur.com/1KGMOpo.png)
![](https://i.imgur.com/SZl59mT.png)

#### ELU_WI_BN
The train loss is **0.1472**
The train accuracy is **94.90%**
The test loss is **0.3007**
The test accuracy is **90.93%**
The run time on GTX1060-6GB is **2:52:39** (h/m/s)
![](https://i.imgur.com/KmsnbTZ.png)
![](https://i.imgur.com/OgqcXf3.png)
![](https://i.imgur.com/IFQJcVn.png)

#### Relu_noWI_BN
The train loss is **0.1351**
The train accuracy is **95.32%**
The test loss is **0.3249**
The test accuracy is **90.71%**
The run time on GTX1060-6GB is **2:52:12** (h/m/s)
![](https://i.imgur.com/VZwe5iJ.png)
![](https://i.imgur.com/UML9Ulv.png)
![](https://i.imgur.com/zIEETMU.png)

#### Relu_noWI_noBN
The train loss is **0.1932**
The train accuracy is **93.24%**
The test loss is **0.3574**
The test accuracy is **89.84%**
The run time on GTX1060-6GB is **1:55:27** (h/m/s)
![](https://i.imgur.com/NFMqjYA.png)
![](https://i.imgur.com/52EjQAb.png)
![](https://i.imgur.com/8uNj9q4.png)


#### Relu_WI_BN
The train loss is **0.1430**
The train accuracy is **95.03%**
The test loss is **0.3334**
The test accuracy is **90.49%**
The run time on GTX1060-6GB is **2:51:42** (h/m/s)
![](https://i.imgur.com/utx2B31.png)
![](https://i.imgur.com/wQR7cYq.png)
![](https://i.imgur.com/B9G1A4o.png)

## Discussion
- Recall the test accuracy
![](https://i.imgur.com/fLZ67fA.png)
### Discussion about Batch Normalization
We know that Batch Normalization accelerate deep network training by reducing internal covariate shift. In this lab, I found that networks that use Batch Normalization are significantly more robust to bad initialization and have higher accuracy. 

Networks with Batch Normalization fit the training data much closerly and also fit the testing data nicer than networks without Batch Normalization.

The cost of Batch Normalization is training time. The training time network with Batch Normalization is one and half longer than the training time network without Batch Normalization.

### Discussion about Weight Initialization
A Neural Network layer that has very small weights will during backpropagation compute very small gradients on its data (since this gradient is proportional to the value of the weights). This could greatly diminish the “gradient signal” flowing backward through a network, and could become a concern for deep networks.

In [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852), He et al. suggest an initialization specifically for ReLU neurons, reaching the conclusion that the variance of neurons in the network should be 2.0/n. 

This gives the initialization `w = np.random.randn(n) * sqrt(2.0/n)`, where `n` is the number of its inputs, and is the current recommendation for use in practice in the specific case of neural networks with ReLU neurons.

However, the effect of Weight Initialization is not clear in the lab.

### Discussion about ReLU v.s. ELU activations

In this lab, ELU activation performed better than ReLU activation. 

[Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)](https://arxiv.org/abs/1511.07289) explain why ELU performence better than ReLU.

> In contrast to ReLUs, ELUs have negative values which allows them to push mean unit activations closer to zero like batch normalization but with lower computational complexity. 

> Mean shifts toward zero speed up learning by bringing the normal gradient closer to the unit natural gradient because of a reduced bias shift effect. While LReLUs and PReLUs have negative values, too, they do not ensure a noise-robust deactivation state. 

> ELUs saturate to a negative value with smaller inputs and thereby decrease the forward propagated variation and information. Therefore, ELUs code the degree of presence of particular phenomena in the input, while they do not quantitatively model the degree of their absence. 

> In experiments, ELUs lead not only to faster learning, but also to significantly better generalization performance than ReLUs and LReLUs on networks with more than 5 layers.
















