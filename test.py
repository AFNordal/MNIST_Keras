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
    model.add(Dense(10,activation='softmax'))
    model.load_weights('weights.h5')
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

net=baseNet()


imfile=open("./displayNet/pics.txt","r")
l=[]
for i in imfile:
    l.append(1-(float(i)/255))


arr=np.asarray(l)
arr=arr.reshape(arr.shape[0]/(784),28,28)
tarr=arr.reshape(arr.shape[0],1,28,28)
print arr.shape[0]
for i in range(int(raw_input("How many images? "))):
    print np.argmax(net.predict_on_batch(tarr)[i])
    pyplot.imshow(arr[i])
    pyplot.show()
pyplot.close()
     