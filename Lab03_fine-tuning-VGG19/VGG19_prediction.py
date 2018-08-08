# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 14:37:15 2017

@author: whisp
"""

from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input, decode_predictions
from keras.models import Model
import numpy as np
import h5py

from PIL import Image
import requests
from io import BytesIO

#url = input('url: ')
url = "http://www.gamers-onlineshop.jp/resize_image.php?image=10301514_56330ae100db5.jpg&width=500"
#url = "http://himg.bdimg.com/sys/portrait/item/48627a5e.jpg"
#url = "https://cdn.pixabay.com/photo/2014/03/29/09/17/cat-300572_1280.jpg"

url = input('url: ')
response = requests.get(url)
img = Image.open(BytesIO(response.content))
print(img.format, img.size, img.mode)


if img.size[1] >= img.size[0]:
    nim = img.crop( (0, int(img.size[1]/2 - img.size[0]/2), img.size[0], int(img.size[1]/2 + img.size[0]/2)) )
else:
    nim = img.crop( (int(img.size[0]/2 - img.size[1]/2), 0, int(img.size[0]/2 + img.size[1]/2), img.size[1]) )

print(nim.format, nim.size, nim.mode)

model = VGG19(weights='imagenet')

img_path = BytesIO(response.content)
img = image.load_img(img_path, target_size=(224, 224))
nim3 = nim.resize( (224, 224), Image.BILINEAR )
x = image.img_to_array(nim3)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features = model.predict(x)
print('Predicted:', decode_predictions(features, top=5)[0])

import gc
gc.collect()
#nim0 = img.copy()
#nim0.thumbnail( (600,600) )
#print(nim0.format, nim0.size, img.mode)
#
#nim2 = img.resize( (32, 32), Image.BILINEAR )
#nim2
#print(nim2.format, nim2.size, img.mode)
#
#nim3 = img.resize( (244, 244), Image.BILINEAR )