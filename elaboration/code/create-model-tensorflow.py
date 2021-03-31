from tensorflow.keras import Model
from tensorflow.keras.layers import Flatten, Dense

class MultiLayerPerceptron(Model):
    def __init__(self):
        super().__init__()
        self.flat = Flatten(input_shape=(28,28))
        self.fc1 = Dense(64, activation='relu')
        self.fc2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.flat(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

batch_size = 50

model = MultiLayerPerceptron()
model.build((batch_size, 28, 28))