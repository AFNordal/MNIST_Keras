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

(x_trn, y_train),(x_tst, y_test)=mnist.load_data()

x_trn=x_trn.astype('float32')
x_tst=x_tst.astype('float32')

x_trn/=255
x_tst/=255



x_train=[]
x_test=[]
for i in x_trn:
    ls=[]
    for j in i:
        for k in j:
            ls.append(k)
    x_train.append(ls)
print "y"

for i in x_tst:
    ls=[]
    for j in i:
        for k in j:
            ls.append(k)
    x_test.append(ls)

x_train=np.asarray(x_train)
x_test=np.asarray(x_test)

print "yo"
net=Sequential()
net.add(Dense(16, input_dim=784, activation='sigmoid'))
net.add(Dense(16, activation='sigmoid'))
net.add(Dense(10, activation='sigmoid'))

net.load_weights('weights.h5')

sgd=optimizers.SGD(lr=0.01)
net.compile(loss='mean_squared_error', optimizer=sgd, metrics=[metrics.categorical_accuracy])
print "yoy"

# x_train=x_train.reshape(x_train.shape[0], 28, 28, 1)
# x_test=x_test.reshape(x_test.shape[0], 28, 28, 1)

y_train=np_utils.to_categorical(y_train, 10)
y_test=np_utils.to_categorical(y_test, 10)
print "yoy"
net.fit(x_train, y_train, epochs=3500, batch_size=200)
net.save_weights('weights.h5')
print "yoyo"
score=net.evaluate(x_test, y_test, batch_size=50)
print score