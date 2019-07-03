import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.layers import Dense, Input
from keras.layers.convolutional import Conv2D
from keras.models import Model

(x_train, y_train), (x_test, y_test) = mnist.load_data()

alpha = 0.6

image_noise = []

for i in range(len(x_train)):
    img = x_train[i,:,:]
    noise = np.random.rand(28,28)
    img  = alpha*img + (1-alpha)*noise*255
    image_noise.append(img)
    
image_noise = np.array(image_noise)


#plt.imshow(image_noise[1,:,:])
#plt.imshow(x_train[1,:,:])


image_noise = np.expand_dims(image_noise, axis = 3)
x_train = np.expand_dims(x_train, axis = 3)


input_img = Input(shape=(28,28,1))
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(input_img)
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv3')(x)
x = Conv2D(1, (3, 3), activation='relu', padding='same', name='block1_conv4')(x)
model = Model(inputs = input_img, outputs = x)

model.summary()

model.compile(optimizer='adam', loss='mse')
model.fit(image_noise, x_train, epochs=20, batch_size=256, shuffle=True)