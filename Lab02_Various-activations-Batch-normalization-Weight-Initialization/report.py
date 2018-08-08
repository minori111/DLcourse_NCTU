# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 13:29:39 2017

@author: whisp
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

'''load data'''
#ELU_noWI_noBN
train_acc_elu = pickle.load( open( "train_acc_elu.txt", "rb" ) )
train_loss_elu = pickle.load( open( "train_loss_elu.txt", "rb" ) )
val_acc_elu = pickle.load( open( "val_acc_elu.txt", "rb" ) )
val_loss_elu = pickle.load( open( "val_loss_elu.txt", "rb" ) )

#ELU_noWI_BN
train_acc_ELU_noWI_BN = pickle.load( open( "train_acc_ELU_noWI_BN.txt", "rb" ) )
train_loss_ELU_noWI_BN = pickle.load( open( "train_loss_ELU_noWI_BN.txt", "rb" ) )
val_acc_ELU_noWI_BN = pickle.load( open( "val_acc_ELU_noWI_BN.txt", "rb" ) )
val_loss_ELU_noWI_BN = pickle.load( open( "val_loss_ELU_noWI_BN.txt", "rb" ) )

#elu_WI_BN
train_acc_elu_WI_BN = pickle.load( open( "train_acc_elu_WI_BN.txt", "rb" ) )
train_loss_elu_WI_BN = pickle.load( open( "train_loss_elu_WI_BN.txt", "rb" ) )
val_acc_elu_WI_BN = pickle.load( open( "val_acc_elu_WI_BN.txt", "rb" ) )
val_loss_elu_WI_BN = pickle.load( open( "val_loss_elu_WI_BN.txt", "rb" ) )

#Relu_noWI_BN
train_acc_Relu_noWI_BN = pickle.load( open( "train_acc_Relu_noWI_BN.txt", "rb" ) )
train_loss_Relu_noWI_BN = pickle.load( open( "train_loss_Relu_noWI_BN.txt", "rb" ) )
val_acc_Relu_noWI_BN = pickle.load( open( "val_acc_Relu_noWI_BN.txt", "rb" ) )
val_loss_Relu_noWI_BN = pickle.load( open( "val_loss_Relu_noWI_BN.txt", "rb" ) )

#Relu_noWI_noBN
train_acc_Relu_noWI_noBN = pickle.load( open( "train_acc_Relu_noWI_noBN.txt", "rb" ) )
train_loss_Relu_noWI_noBN = pickle.load( open( "train_loss_Relu_noWI_noBN.txt", "rb" ) )
val_acc_Relu_noWI_noBN = pickle.load( open( "val_acc_Relu_noWI_noBN.txt", "rb" ) )
val_loss_Relu_noWI_noBN = pickle.load( open( "val_loss_Relu_noWI_noBN.txt", "rb" ) )

#Relu_WI_BN
train_acc_Relu_WI_BN = pickle.load( open( "train_acc_Relu_WI_BN.txt", "rb" ) )
train_loss_Relu_WI_BN = pickle.load( open( "train_loss_Relu_WI_BN.txt", "rb" ) )
val_acc_Relu_WI_BN = pickle.load( open( "val_acc_Relu_WI_BN.txt", "rb" ) )
val_loss_Relu_WI_BN = pickle.load( open( "val_loss_Relu_WI_BN.txt", "rb" ) )

#plot val_loss graph and save it
plt.figure()
plt.plot(range(1, 165), val_loss_elu, color='blue', label='val_loss_ELU_noWI_noBN')
plt.plot(range(1, 165), val_loss_ELU_noWI_BN, color='green', label='val_loss_ELU_noWI_BN')
plt.plot(range(1, 165), val_loss_elu_WI_BN, color='red', label='val_loss_ELU_WI_BN')
plt.plot(range(1, 165), val_loss_Relu_noWI_BN, color='cyan', label='val_loss_Relu_noWI_BN')
plt.plot(range(1, 165), val_loss_Relu_noWI_noBN, color='black', label='val_loss_Relu_noWI_noBN')
plt.plot(range(1, 165), val_loss_Relu_WI_BN, color='yellow', label='val_loss_Relu_WI_BN')
plt.legend(loc="upper right")
plt.xlabel('#Epoch')
plt.ylabel('Loss')
plt.savefig('fig-NIN-all-val_loss.png', dpi=1200)

#plot train_loss graph and save it
plt.figure()
plt.plot(range(1, 165), train_loss_elu, color='blue', label='train_loss_ELU_noWI_noBN')
plt.plot(range(1, 165), train_loss_ELU_noWI_BN, color='green', label='train_loss_ELU_noWI_BN')
plt.plot(range(1, 165), train_loss_elu_WI_BN, color='red', label='train_loss_ELU_WI_BN')
plt.plot(range(1, 165), train_loss_Relu_noWI_BN, color='cyan', label='train_loss_Relu_noWI_BN')
plt.plot(range(1, 165), train_loss_Relu_noWI_noBN, color='black', label='train_loss_Relu_noWI_noBN')
plt.plot(range(1, 165), train_loss_Relu_WI_BN, color='yellow', label='train_loss_Relu_WI_BN')
plt.legend(loc="upper right")
plt.xlabel('#Epoch')
plt.ylabel('Loss')
plt.savefig('fig-NIN-all-train_loss.png', dpi=1200)

#plot val_acc graph and save it
plt.figure()
plt.plot(range(1, 165), val_acc_elu, color='blue', label='val_acc_ELU_noWI_noBN')
plt.plot(range(1, 165), val_acc_ELU_noWI_BN, color='green', label='val_acc_ELU_noWI_BN')
plt.plot(range(1, 165), val_acc_elu_WI_BN, color='red', label='val_acc_ELU_WI_BN')
plt.plot(range(1, 165), val_acc_Relu_noWI_BN, color='cyan', label='val_acc_Relu_noWI_BN')
plt.plot(range(1, 165), val_acc_Relu_noWI_noBN, color='black', label='val_acc_Relu_noWI_noBN')
plt.plot(range(1, 165), val_acc_Relu_WI_BN, color='yellow', label='val_acc_Relu_WI_BN')
plt.legend(loc="lower right")
plt.xlabel('#Epoch')
plt.ylabel('Loss')
plt.savefig('fig-NIN-all-val_acc.png', dpi=1200)

#plot train_acc graph and save it
plt.figure()
plt.plot(range(1, 165), train_acc_elu, color='blue', label='train_acc_ELU_noWI_noBN')
plt.plot(range(1, 165), train_acc_ELU_noWI_BN, color='green', label='train_acc_ELU_noWI_BN')
plt.plot(range(1, 165), train_acc_elu_WI_BN, color='red', label='train_acc_ELU_WI_BN')
plt.plot(range(1, 165), train_acc_Relu_noWI_BN, color='cyan', label='train_acc_Relu_noWI_BN')
plt.plot(range(1, 165), train_acc_Relu_noWI_noBN, color='black', label='train_acc_Relu_noWI_noBN')
plt.plot(range(1, 165), train_acc_Relu_WI_BN, color='yellow', label='train_acc_Relu_WI_BN')
plt.legend(loc="lower right")
plt.xlabel('#Epoch')
plt.ylabel('Loss')
plt.savefig('fig-NIN-all-train_acc.png', dpi=1200)


#plot val_loss and val_acc graph and save it
plt.figure()
plt.plot(range(1, 165), val_loss_elu, color='blue', label='val_loss_ELU_noWI_noBN')
plt.plot(range(1, 165), val_loss_ELU_noWI_BN, color='green', label='val_loss_ELU_noWI_BN')
plt.plot(range(1, 165), val_loss_elu_WI_BN, color='red', label='val_loss_ELU_WI_BN')
plt.plot(range(1, 165), val_loss_Relu_noWI_BN, color='cyan', label='val_loss_Relu_noWI_BN')
plt.plot(range(1, 165), val_loss_Relu_noWI_noBN, color='black', label='val_loss_Relu_noWI_noBN')
plt.plot(range(1, 165), val_loss_Relu_WI_BN, color='yellow', label='val_loss_Relu_WI_BN')

plt.plot(range(1, 165), val_acc_elu, color='blue', label='val_acc_ELU_noWI_noBN')
plt.plot(range(1, 165), val_acc_ELU_noWI_BN, color='green', label='val_acc_ELU_noWI_BN')
plt.plot(range(1, 165), val_acc_elu_WI_BN, color='red', label='val_acc_ELU_WI_BN')
plt.plot(range(1, 165), val_acc_Relu_noWI_BN, color='cyan', label='val_acc_Relu_noWI_BN')
plt.plot(range(1, 165), val_acc_Relu_noWI_noBN, color='black', label='val_acc_Relu_noWI_noBN')
plt.plot(range(1, 165), val_acc_Relu_WI_BN, color='yellow', label='val_acc_Relu_WI_BN')
plt.legend(loc="center right")
plt.xlabel('#Epoch')
plt.ylabel('Loss')
plt.savefig('fig-NIN-all-val_loss-val_acc.png', dpi=1200)






















