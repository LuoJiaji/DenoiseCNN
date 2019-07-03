import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.layers import Dense, Input
from keras.layers.convolutional import Conv2D
from keras.models import Model, load_model
from keras.optimizers import SGD

# 加载数据
(x_train, y_train), (x_test, y_test) = mnist.load_data()
alpha = 0.6
image_noise = []

for i in range(len(x_train)):
    img = x_train[i,:,:]
    noise = np.random.rand(28,28)
    img  = alpha*img + (1-alpha)*noise*255
    image_noise.append(img)
    
image_noise = np.array(image_noise)

#plt.figure()
#plt.imshow(image_noise[1,:,:])
#plt.figure()
#plt.imshow(x_train[1,:,:])

image_noise = np.expand_dims(image_noise, axis = 3)
x_train = np.expand_dims(x_train, axis = 3)

# 定义模型
input_img = Input(shape=(28,28,1))
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(input_img)
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv3')(x)
x = Conv2D(1, (3, 3), activation='relu', padding='same', name='block1_conv4')(x)
model = Model(inputs = input_img, outputs = x)
model.summary()

# 模型训练
sgd = SGD(lr=0.1, decay=1e-6)   
model.compile(optimizer=sgd, loss='mse')
model.fit(image_noise, x_train, epochs=20, batch_size=256, shuffle=True)

# 模型的保存于加载
#model.save('./models/DenoiseCNN.h5')
model = load_model('./models/DenoiseCNN.h5')

# 在测试集中测试
ind = 3000
img = x_test[ind,:,:]
alpha = 0.4
noise = np.random.rand(28,28)
image_noise  = alpha*img + (1-alpha)*noise*255
image_noise = np.expand_dims(image_noise, axis = 0)
image_noise = np.expand_dims(image_noise, axis = 3)

image_denoise = model.predict(image_noise)

#plt.figure()
plt.subplot(2,2,1)
plt.title('image noise')
plt.imshow(image_noise[0,:,:,0])
plt.axis('off')
#plt.figure()
plt.subplot(2,2,2)
plt.title('image original')
plt.imshow(img)
plt.axis('off')
#plt.figure()
plt.subplot(2,2,3)
plt.title('image denoise')
plt.imshow(image_denoise[0,:,:,0])
plt.axis('off')

# 测试二次滤波效果
image_denoise_2 = model.predict(image_denoise)

plt.subplot(2,2,4)
plt.title('image denoise*2')
plt.imshow(image_denoise_2[0,:,:,0])
plt.axis('off')

