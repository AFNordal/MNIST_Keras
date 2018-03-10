from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras import metrics
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
from matplotlib import pyplot
from keras.models import load_model
from keras.models import save_model
np.random.seed(7)

net=Sequential()
net.add(Dense(16, input_dim=784, activation='sigmoid'))
net.add(Dense(16, activation='sigmoid'))
net.add(Dense(10, activation='sigmoid'))

net.load_weights('weights.h5')

sgd=optimizers.SGD(lr=0.01)
net.compile(loss='mean_squared_error', optimizer=sgd, metrics=[metrics.categorical_accuracy])


imfile=open("./displayNet/displayNet/pics.txt","r")
l=[]
for i in imfile:
    l.append(1-(float(i)/255))

xl=[]
for i in range(len(l)/784):
    ls=[]
    for j in range(784):
        ls.append(l[784*i+j])
    xl.append(ls)


tarr=np.asarray(xl)
print tarr
arr=np.asarray(l)
arr=arr.reshape(arr.shape[0]/(784),28,28)
for i in range(int(raw_input("How many images? "))):
    print np.argmax(net.predict_on_batch(tarr)[i])
    pyplot.imshow(arr[i])
    pyplot.show()
print arr.shape
pyplot.close()
     