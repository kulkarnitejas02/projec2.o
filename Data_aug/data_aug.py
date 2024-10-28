# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 15:05:39 2023

"""

#### Data Augmentation

# Importing necessary functions
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# Initialising the ImageDataGenerator class.
# We will pass in the augmentation parameters in the constructor.
datagen = ImageDataGenerator(
    rotation_range = 45,
    zoom_range = 0.4,
    shear_range = 0.2,
    horizontal_flip = True,
	brightness_range = (0.5, 2.5))

# Loading a sample image
img = load_img('Black_6.jpg')

# Converting the input sample image to an array
x = img_to_array(img)

# Reshaping the input image
x = x.reshape((1, ) + x.shape)

# Generating and saving 5 augmented samples
# using the above defined parameters.
i = 0
for batch in datagen.flow(x, batch_size = 1,
						save_to_dir ='preview',
						save_prefix ='image', save_format ='jpeg'):
	i += 1
	if i > 5:
		break
