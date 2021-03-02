import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model

xception_no_top = tf.keras.applications.Xception(include_top=False, 
                    input_shape=(299, 299, 3))
x = xception_no_top.layers[-1].output
flat = Flatten()(x)
output = Dense(units=len(classes), activation='softmax')(flat)

model = Model(inputs=xception_no_top.input, outputs=output)
for layer in model.layers[:-10]:
    layer.trainable = False