from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np


(x_train, y_train), (x_test, y_test) = mnist.load_data()



alpha = 0.6

image_noise = []

for i in range(len(x_train)):
    img = x_train[i,:,:]
    noise = np.random.rand(28,28)
    img  = alpha*img + (1-alpha)*noise*255
    image_noise.append(img)
    
image_noise = np.array(image_noise)

plt.imshow(image_noise[1,:,:])
plt.imshow(x_train[1,:,:])