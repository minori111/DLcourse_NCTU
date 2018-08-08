# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 00:42:16 2017

@author: whisp
"""

import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib
matplotlib.style.use('bmh')


'''load data'''
#vgg19 pretrain
train_acc_with_pretrain = pickle.load( open( "train_acc_pretrain.txt", "rb" ) )
train_loss_with_pretrain = pickle.load( open( "train_loss_pretrain.txt", "rb" ) )
val_acc_with_pretrain = pickle.load( open( "val_acc_pretrain.txt", "rb" ) )
val_loss_with_pretrain = pickle.load( open( "val_loss_pretrain.txt", "rb" ) )

#no pretrain
train_acc_pretrain_no_weight = pickle.load( open( "train_acc_pretrain_no_weight.txt", "rb" ) )
train_loss_pretrain_no_weight = pickle.load( open( "train_loss_pretrain_no_weight.txt", "rb" ) )
val_acc_pretrain_no_weight = pickle.load( open( "val_acc_pretrain_no_weight.txt", "rb" ) )
val_loss_pretrain_no_weight = pickle.load( open( "val_loss_pretrain_no_weight.txt", "rb" ) )

#vgg19 pretrain BN
train_acc_pretrain_weight_he_normal = pickle.load( open( "train_acc_pretrain_weight_he_normal.txt", "rb" ) )
train_loss_pretrain_weight_he_normal = pickle.load( open( "train_loss_pretrain_weight_he_normal.txt", "rb" ) )
val_acc_pretrain_weight_he_normal = pickle.load( open( "val_acc_pretrain_weight_he_normal.txt", "rb" ) )
val_loss_pretrain_weight_he_normal = pickle.load( open( "val_loss_pretrain_weight_he_normal.txt", "rb" ) )


'''plot group curve'''
#plot train_acc and val_acc with_pretrain_BN graph and save it
plt.figure()
plt.plot(range(1, 165), train_acc_pretrain_weight_he_normal, color='blue', label='train_acc_pretrain_weight_he_normal')
plt.plot(range(1, 165), val_acc_pretrain_weight_he_normal, color='red', label='val_acc_pretrain_weight_he_normal')
plt.legend(loc="lower right")
plt.xlabel('#Epoch')
plt.ylabel('Loss')
plt.savefig('with_pretrain_BN_acc.png', dpi=1200)

#plot train_loss and val_loss with_pretrain_BN graph and save it
plt.figure()
plt.plot(range(1, 165), train_loss_pretrain_weight_he_normal, color='blue', label='train_loss_pretrain_weight_he_normal')
plt.plot(range(1, 165), val_loss_pretrain_weight_he_normal, color='red', label='val_loss_pretrain_weight_he_normal')
plt.legend(loc="upper right")
plt.xlabel('#Epoch')
plt.ylabel('Loss')
plt.savefig('with_pretrain_BN_loss.png', dpi=1200)

#plot train_acc and val_acc with_pretrain graph and save it
plt.figure()
plt.plot(range(1, 165), train_acc_with_pretrain, color='blue', label='train_acc_with_pretrain')
plt.plot(range(1, 165), val_acc_with_pretrain, color='red', label='val_acc_with_pretrain')
plt.legend(loc="lower right")
plt.xlabel('#Epoch')
plt.ylabel('Loss')
plt.savefig('with_pretrain_acc.png', dpi=1200)

#plot train_loss and val_loss with_pretrain graph and save it
plt.figure()
plt.plot(range(1, 165), train_loss_with_pretrain, color='blue', label='train_loss_with_pretrain')
plt.plot(range(1, 165), val_loss_with_pretrain, color='red', label='val_loss_with_pretrain')
plt.legend(loc="upper right")
plt.xlabel('#Epoch')
plt.ylabel('Loss')
plt.savefig('with_pretrain_loss.png', dpi=1200)


#plot train_acc and val_acc with_no_pretrain graph and save it
plt.figure()
plt.plot(range(1, 165), train_acc_pretrain_no_weight, color='blue', label='train_acc_pretrain_no_weight')
plt.plot(range(1, 165), val_acc_pretrain_no_weight, color='red', label='val_acc_pretrain_no_weight')
plt.legend(loc="lower right")
plt.xlabel('#Epoch')
plt.ylabel('Loss')
plt.savefig('with_no_pretrain_acc.png', dpi=1200)

#plot train_loss and val_loss with_no_pretrain graph and save it
plt.figure()
plt.plot(range(1, 165), train_loss_pretrain_no_weight, color='blue', label='train_loss_pretrain_no_weight')
plt.plot(range(1, 165), val_loss_pretrain_no_weight, color='red', label='val_loss_pretrain_no_weight')
plt.legend(loc="upper right")
plt.xlabel('#Epoch')
plt.ylabel('Loss')
plt.savefig('with_pretrain_no_loss.png', dpi=1200)



'''plot comparsion curve'''
#plot val_loss graph and save it
plt.figure()
plt.plot(range(1, 165), val_loss_with_pretrain, color='blue', label='val_loss_with_pretrain')
plt.plot(range(1, 165), val_loss_pretrain_no_weight, color='red', label='val_loss_pretrain_no_weight')
plt.plot(range(1, 165), val_loss_pretrain_weight_he_normal, color='green', label='val_loss_pretrain_weight_he_normal')
plt.legend(loc="upper right")
plt.xlabel('#Epoch')
plt.ylabel('Loss')
plt.savefig('val_loss.png', dpi=1200)


#plot train_loss graph and save it
plt.figure()
plt.plot(range(1, 165), train_loss_with_pretrain, color='blue', label='train_loss_with_pretrain')
plt.plot(range(1, 165), train_loss_pretrain_no_weight, color='red', label='train_loss_pretrain_no_weight')
plt.plot(range(1, 165), train_loss_pretrain_weight_he_normal, color='green', label='train_loss_pretrain_weight_he_normal')
plt.legend(loc="upper right")
plt.xlabel('#Epoch')
plt.ylabel('Loss')
plt.savefig('train_loss.png', dpi=1200)

#plot val_acc graph and save it
plt.figure()
plt.plot(range(1, 165), val_acc_with_pretrain, color='blue', label='val_acc_with_pretrain')
plt.plot(range(1, 165), val_acc_pretrain_no_weight, color='red', label='val_acc_pretrain_no_weight')
plt.plot(range(1, 165), val_acc_pretrain_weight_he_normal, color='green', label='val_acc_pretrain_weight_he_normal')
plt.legend(loc="lower right")
plt.xlabel('#Epoch')
plt.ylabel('Loss')
plt.savefig('val_acc.png', dpi=1200)

#plot train_acc graph and save it
plt.figure()
plt.plot(range(1, 165), train_acc_with_pretrain, color='blue', label='train_acc_with_pretrain')
plt.plot(range(1, 165), train_acc_pretrain_no_weight, color='red', label='train_acc_pretrain_no_weight')
plt.plot(range(1, 165), train_acc_pretrain_weight_he_normal, color='green', label='train_acc_pretrain_weight_he_normal')
plt.legend(loc="lower right")
plt.xlabel('#Epoch')
plt.ylabel('Loss')
plt.savefig('train_acc.png', dpi=1200)


#plot val_loss and val_acc graph and save it
#plt.figure()
#plt.plot(range(1, 165), val_loss_elu, color='blue', label='val_loss_ELU_noWI_noBN')
#plt.plot(range(1, 165), val_loss_ELU_noWI_BN, color='red', label='val_loss_ELU_noWI_BN')
#plt.plot(range(1, 165), val_loss_elu_WI_BN, color='red', label='val_loss_ELU_WI_BN')
#plt.plot(range(1, 165), val_loss_Relu_noWI_BN, color='cyan', label='val_loss_Relu_noWI_BN')
#plt.plot(range(1, 165), val_loss_Relu_noWI_noBN, color='black', label='val_loss_Relu_noWI_noBN')
#plt.plot(range(1, 165), val_loss_Relu_WI_BN, color='yellow', label='val_loss_Relu_WI_BN')
#
#plt.plot(range(1, 165), val_acc_elu, color='blue', label='val_acc_ELU_noWI_noBN')
#plt.plot(range(1, 165), val_acc_ELU_noWI_BN, color='red', label='val_acc_ELU_noWI_BN')
#plt.plot(range(1, 165), val_acc_elu_WI_BN, color='red', label='val_acc_ELU_WI_BN')
#plt.plot(range(1, 165), val_acc_Relu_noWI_BN, color='cyan', label='val_acc_Relu_noWI_BN')
#plt.plot(range(1, 165), val_acc_Relu_noWI_noBN, color='black', label='val_acc_Relu_noWI_noBN')
#plt.plot(range(1, 165), val_acc_Relu_WI_BN, color='yellow', label='val_acc_Relu_WI_BN')
#plt.legend(loc="center right")
#plt.xlabel('#Epoch')
#plt.ylabel('Loss')
#plt.savefig('val_loss-val_acc.png', dpi=1200)








