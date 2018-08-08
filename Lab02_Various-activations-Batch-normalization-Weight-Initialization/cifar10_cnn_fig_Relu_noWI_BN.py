'''Train a simple deep CNN on the CIFAR10 small images dataset.

GPU run command with Theano backend (with TensorFlow, the GPU is automatically used):
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatx=float32 python cifar10_cnn.py

It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras import optimizers
from keras.callbacks import LearningRateScheduler
from keras.initializers import RandomNormal, he_normal
from keras.layers.advanced_activations import ELU



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

#

# learning rate schedule
def step_decay(epoch):
    lrate = 0.1
    if epoch > 81:
        lrate = 0.01
    if epoch > 121:
        lrate = 0.001
    return lrate
#
#
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

model = Sequential()
m = 0.0
sd = 0.05

#use_bias = True, bias_initializer = RandomNormal(mean=0.05, stddev=0.05, seed=None),
model.add(Conv2D(192, (5, 5), padding='same',   use_bias = True, bias_initializer = RandomNormal(mean=m, stddev=0.01, seed=None),                
                 input_shape=x_train.shape[1:]))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
model.add(Activation('relu'))
model.add(Conv2D(160, (1, 1),   use_bias = True, bias_initializer = RandomNormal(mean=m, stddev=sd, seed=None)))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
model.add(Activation('relu'))
model.add(Conv2D(96, (1, 1),   use_bias = True, bias_initializer = RandomNormal(mean=m, stddev=sd, seed=None)))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides = 2))
model.add(Dropout(0.5))

model.add(Conv2D(192, (5, 5), padding='same',   use_bias = True, bias_initializer = RandomNormal(mean=m, stddev=sd, seed=None)))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
model.add(Activation('relu'))
model.add(Conv2D(192, (1, 1),   use_bias = True, bias_initializer = RandomNormal(mean=m, stddev=sd, seed=None)))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
model.add(Activation('relu'))
model.add(Conv2D(192, (1, 1),   use_bias = True, bias_initializer = RandomNormal(mean=m, stddev=sd, seed=None)))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides = 2))
model.add(Dropout(0.5))

model.add(Conv2D(192, (3, 3), padding='same',   use_bias = True, bias_initializer = RandomNormal(mean=m, stddev=sd, seed=None)))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
model.add(Activation('relu'))
model.add(Conv2D(192, (1, 1),   use_bias = True, bias_initializer = RandomNormal(mean=m, stddev=sd, seed=None)))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
model.add(Activation('relu'))
model.add(Conv2D(10, (1, 1),   use_bias = True, bias_initializer = RandomNormal(mean=m, stddev=sd, seed=None)))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
model.add(Activation('relu'))
model.add(GlobalAveragePooling2D())
#model.add(Flatten())
#model.add(Dense(512))
#model.add(Activation('relu'))
#model.add(Dropout(0.5))
#model.add(Dense(num_classes))
model.add(Activation('softmax'))


sgd = optimizers.SGD(lr=0.1, decay=0.0, momentum=0.75, 
                     nesterov=True)
# Let's train the model using sgd
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
lrate = LearningRateScheduler(step_decay)
callbacks_list = [lrate]


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#Normalize each color channel
x_train[:,:,:,0] = (x_train[:,:,:,0]-125.3)/63.0
x_train[:,:,:,1] = (x_train[:,:,:,1]-123.3)/62.1
x_train[:,:,:,2] = (x_train[:,:,:,2]-113.9)/66.7
#x_test /= 255
x_test[:,:,:,0] = (x_test[:,:,:,0]-125.3)/63.0
x_test[:,:,:,1] = (x_test[:,:,:,1]-123.3)/62.1
x_test[:,:,:,2] = (x_test[:,:,:,2]-113.9)/66.7


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
    
    #save reult to txt
    pickle.dump( train_loss, open( "train_loss_Relu_noWI_BN.txt", "wb" ) )
    pickle.dump( train_acc, open( "train_acc_Relu_noWI_BN.txt", "wb" ) )
    pickle.dump( val_loss, open( "val_loss_Relu_noWI_BN.txt", "wb" ) )
    pickle.dump( val_acc, open( "val_acc_Relu_noWI_BN.txt", "wb" ) )
    
    #plot loss graph and save it
    plt.plot(range(1, len(train_loss)+1), train_loss, color='blue', label='Train loss')
    plt.plot(range(1, len(val_loss)+1), val_loss, color='red', label='Val loss')
    plt.legend(loc="upper right")
    plt.xlabel('#Epoch')
    plt.ylabel('Loss')
    plt.savefig('fig-NIN-loss_Relu_noWI_BN.png', dpi=1200)
    
    
    #plot acc graph and save it
    plt.figure()
    plt.plot(range(1, len(train_acc)+1), train_acc, color='yellow', label='Train acc')
    plt.plot(range(1, len(val_acc)+1), val_acc, color='green', label='Val acc')
    plt.legend(loc="upper right")
    plt.xlabel('#Epoch')
    plt.ylabel('Loss')
    plt.savefig('fig-NIN-acc_Relu_noWI_BN.png', dpi=1200)
    
import gc
gc.collect()
#plt.plot(range(1, len(train_loss)+1), train_loss, color='blue', label='Train loss')
#plt.plot(range(1, len(val_loss)+1), val_loss, color='red', label='Val loss')
#plt.legend(loc="upper right")
#plt.xlabel('#Epoch')
#plt.ylabel('Loss')
#plt.savefig('fig-NIN-loss.png', dpi=1200)
#
#plt.figure()
#plt.plot(range(1, len(train_acc)+1), train_acc, color='yellow', label='Train acc')
#plt.plot(range(1, len(val_acc)+1), val_acc, color='green', label='Val acc')
#plt.legend(loc="upper right")
#plt.xlabel('#Epoch')
#plt.ylabel('Loss')
#plt.savefig('fig-NIN-acc.png', dpi=1200)
