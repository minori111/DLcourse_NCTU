# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 19:31:01 2017

@author: whisp
"""
import keras
from keras.utils.data_utils import get_file
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.datasets import cifar10
from keras import optimizers
from keras.callbacks import LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from keras.initializers import RandomNormal
import numpy as np

import pickle
import matplotlib.pyplot as plt

# inline plotting instead of popping out
#%matplotlib inline

# load utility classes/functions that has been taught in previous labs
# e.g., plot_decision_regions()
import datetime
import time
import os, sys
module_path = os.path.abspath(os.path.join('.'))
sys.path.append(module_path)


# learning rate schedule
def step_decay(epoch):
    lrate = 0.01
    if epoch > 81:
        lrate = 0.001
    if epoch > 121:
        lrate = 0.0001
    return lrate

#set up parameters
batch_size = 128
num_classes = 10
epochs = 164
#data_augmentation = False
data_augmentation = True

# input image dimensions
img_rows, img_cols = 32, 32
# The CIFAR10 images are RGB.
img_channels = 3

# The data, shuffled and split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


#vgg19-load weight path
WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels.h5'
filepath = get_file('vgg19_weights_tf_dim_ordering_tf_kernels.h5', WEIGHTS_PATH, cache_subdir='models')

#build model
model = Sequential()
# Block 1
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', input_shape=x_train.shape[1:]))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

# Block 2
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

# Block 3
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

# Block 4
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

# Block 5
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4'))

#Some modification of VGG19
model.add(Flatten(name='flatten'))
model.add(Dense(4096, use_bias = True, bias_initializer = RandomNormal(mean=0.0, stddev=0.01), activation='relu', name='fc_new'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu', name='fc2'))  
model.add(Dropout(0.5))      
model.add(Dense(10, activation='softmax', name='predictions_new'))        

#load pretrained weight from VGG19 by name      
model.load_weights(filepath, by_name=True)




#set-up optimizer
sgd = optimizers.SGD(lr=0.1, decay=0.0, momentum=0.75, 
                     nesterov=True)
# Let's train the model using sgd
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
#use schedulered learning rate
lrate = LearningRateScheduler(step_decay)
callbacks_list = [lrate]


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#Normalize each color channel x_train
x_train[:,:,:,0] = (x_train[:,:,:,0]-123.68)
x_train[:,:,:,1] = (x_train[:,:,:,1]-116.779)
x_train[:,:,:,2] = (x_train[:,:,:,2]-103.939)
#x_test
x_test[:,:,:,0] = (x_test[:,:,:,0]-123.68)
x_test[:,:,:,1] = (x_test[:,:,:,1]-116.779)
x_test[:,:,:,2] = (x_test[:,:,:,2]-103.939)


#fit model
if not data_augmentation:
    print('Not using data augmentation.')
    start_time = time.clock()
    his = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=False, 
              callbacks=callbacks_list)
    print('The run time is '+str(datetime.timedelta(seconds = (time.clock()-start_time))))
    train_loss = his.history['loss']
    train_acc = his.history['acc']
    val_loss = his.history['val_loss']
    val_acc = his.history['val_acc']

else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    start_time = time.clock()
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    
    his = model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        steps_per_epoch=x_train.shape[0] // batch_size,
                        epochs=epochs,
                        validation_data=(x_test, y_test), 
                        callbacks=callbacks_list)
    print('The run time is '+str(datetime.timedelta(seconds = (time.clock()-start_time))))
    
    #read the result
    train_loss = his.history['loss']
    train_acc = his.history['acc']
    val_loss = his.history['val_loss']
    val_acc = his.history['val_acc']
    
    #save result to txt
    pickle.dump( train_loss, open( "train_loss_pretrain_weight_v1.txt", "wb" ) )
    pickle.dump( train_acc, open( "train_acc_pretrain_weight_v1.txt", "wb" ) )
    pickle.dump( val_loss, open( "val_loss_pretrain_weight_v1.txt", "wb" ) )
    pickle.dump( val_acc, open( "val_acc_pretrain_weight_v1.txt", "wb" ) )
    
    #plot loss graph and save it
    plt.plot(range(1, len(train_loss)+1), train_loss, color='blue', label='Train loss')
    plt.plot(range(1, len(val_loss)+1), val_loss, color='red', label='Val loss')
    plt.legend(loc="upper right")
    plt.xlabel('#Epoch')
    plt.ylabel('Loss')
    plt.savefig('pre_train_VGG19_cifa10_loss_v1.png', dpi=1200)
    
    
    #plot acc graph and save it
    plt.figure()
    plt.plot(range(1, len(train_acc)+1), train_acc, color='yellow', label='Train acc')
    plt.plot(range(1, len(val_acc)+1), val_acc, color='green', label='Val acc')
    plt.legend(loc="upper right")
    plt.xlabel('#Epoch')
    plt.ylabel('Loss')
    plt.savefig('pre_train_VGG19_cifa10_acc_v1.png', dpi=1200)

#close status    
import gc
gc.collect()













        
