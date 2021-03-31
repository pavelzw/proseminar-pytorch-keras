import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.models import Model

inp = Input(shape=(image_size, image_size, 3))
h = Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(inp)
h = MaxPool2D(pool_size=(2,2), strides=2)(h)
h = Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same')(h)
h = MaxPool2D(pool_size=(2,2), strides=2)(h)
h = Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same')(h)

h = Flatten()(h)
outputs = Dense(units=2, activation='softmax')(h)

model = Model(inputs=inputs, outputs=outputs)
