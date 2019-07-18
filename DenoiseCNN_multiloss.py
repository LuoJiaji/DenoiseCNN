import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model, load_model
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from keras.layers.normalization import BatchNormalization

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
x_test = np.expand_dims(x_test, axis = 3)

x_train = x_train.astype('float32')
image_noise = image_noise.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)



input_img = Input(shape=(28,28,1))
x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv1')(input_img)
x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
#x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv3')(x)
re_out = Conv2D(1, (3, 3), activation='relu', padding='same', name='re_out')(x)

x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool1')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool2')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv3')(x)
x = BatchNormalization(name= 'bn_block1_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool3')(x)
x = Flatten(name='flatten')(x)
x = Dense(128, activation='relu', name='fc1')(x)
x = Dropout(0.5)(x)
x = Dense(10, activation='softmax', name='fc_out')(x)


#multiloss 模型
model_all = Model(inputs = input_img, outputs = [re_out,x])
model_all.compile(optimizer=SGD(), 
              loss={'re_out': 'mse', 'fc_out': 'categorical_crossentropy'}, 
              loss_weights={'re_out': 1.,'fc_out': 1.},
              metrics={'re_out':'mae','fc_out':'accuracy'})
model_all.summary()
plot_model(model_all, to_file='./model_visualization/DenoiseCNN_multiloss.png',show_shapes=True)
model_all.fit(image_noise, [x_train, y_train], epochs=40, batch_size=64, shuffle=True)



#singleloss 分类模型
model_cl = Model(inputs = input_img, outputs = x)
model_cl.compile(optimizer=SGD(), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
model_cl.summary()
plot_model(model_cl, to_file='./model_visualization/DenoiseCNN_classify.png', show_shapes=True)
model_cl.fit(image_noise, y_train, epochs=40, batch_size=64, shuffle=True)



#singleloss 模型
model_re = Model(inputs = input_img, outputs = re_out)
model_re.compile(optimizer=SGD(), 
              loss='mse')
model_re.summary()
plot_model(model_re, to_file='./model_visualization/DenoiseCNN_reconstruction.png', show_shapes=True)
model_re.fit(image_noise, x_train, epochs=40, batch_size=64, shuffle=True)


#tmp = x_test[1]
#tmp = np.expand_dims(tmp, axis = 0)
#re = model.predict(tmp)

