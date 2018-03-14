from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras import backend as K
K.set_image_dim_ordering('th')
from keras import optimizers
from keras import metrics
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
from matplotlib import pyplot
from keras.models import load_model
from keras.models import save_model
np.random.seed(7)

(x_trn, y_trn),(x_tst, y_tst)=mnist.load_data()

x_trn=x_trn.reshape(x_trn.shape[0],1,28,28).astype('float32')
x_tst=x_tst.reshape(x_tst.shape[0],1,28,28).astype('float32')

x_trn/=255
x_tst/=255

y_trn=np_utils.to_categorical(y_trn, 10)
y_tst=np_utils.to_categorical(y_tst, 10)

def baseNet():
    model=Sequential()
    model.add(Conv2D(30,(5,5),input_shape=(1,28,28),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(15,(3,3),input_shape=(1,28,28),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(Dense(50,activation='relu'))
    model.add(Dense(y_tst.shape[1],activation='softmax'))
    model.load_weights('weights.h5')
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

net=baseNet()
net.fit(x_trn,y_trn,validation_data=(x_tst,y_tst),epochs=3,batch_size=200,verbose=1)

net.save_weights('weights.h5')

score=net.evaluate(x_tst,y_tst,verbose=1)
print score











