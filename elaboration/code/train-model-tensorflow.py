from tensorflow.keras.optimizers import SGD

model.compile(optimizer=SGD(learning_rate=0.01),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

batch_size=50

model.fit(x=x_train, y=y_train, epochs=3, validation_data=(x_test, y_test), 
          batch_size=batch_size, verbose=2)
