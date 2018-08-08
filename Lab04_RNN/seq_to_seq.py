# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 04:38:57 2017

@author: whisp
"""
import string
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


#generate random input string
char = string.ascii_uppercase + string.digits + string.ascii_lowercase + string.punctuation
char = string.printable
input_size = 20
def id_generator(size=input_size, chars=char):
    return ''.join(random.choice(chars) for _ in range(size))

st = id_generator()
print(st)

# Convert string to binary format
def binary_format(st):
    return ' '.join('{0:08b}'.format(ord(x), 'b') for x in st)

st_ = binary_format(st)
print(st_)

# Convert char to dataframe
def convert_dataframe(st_):
    df1 = pd.DataFrame()
    for i in range(input_size):
        df1[i] = np.array(list(st_.split(" ")[i])).astype(np.int)
    return df1


df1 = convert_dataframe(st_)
# Set up the matplotlib figure
mask = np.zeros_like(df1, dtype=np.bool)
f, ax = plt.subplots(figsize=(11, 9))
# Draw the heatmap with the mask and input bits
sns.heatmap(df1, mask=mask, cmap="YlGnBu", vmax=1.5, vmin=-0.1,
            square=True, xticklabels=5, yticklabels=5,
            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)



