import numpy as np
import pandas as pd
from keras.applications.xception import Xception
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
import keras

print('Loading data..')
data1=np.load('data/data_save0.npy',allow_pickle=True)
data2=np.load('data/data_save1.npy',allow_pickle=True)

data=[]
labels=[]
for i in range(0,10000):
    data.append(data1[i][0])
    labels.append(data1[i][1])
for i in range(0,10000):
    data.append(data2[i][0])
    labels.append(data2[i][1])

data=np.array(data)
labels=np.array(labels)

print('Data shape is '+ str(data.shape))
print('Label shape is '+ str(labels.shape))

base_model = Xception(weights=None, include_top=False, input_shape=(200, 256, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(8, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)
learning_rate = 0.001
opt = keras.optimizers.adam(lr=learning_rate, decay=1e-5)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


X_train,X_test,Y_train,Y_test= train_test_split(data,labels,test_size=0.2)

print(str(np.shape(X_train)))
print(str(np.shape(X_test)))
print(str(np.shape(Y_train)))
print(str(np.shape(Y_test)))

print('Model loaded..')
print('Now training...')

history=model.fit(X_train,Y_train, verbose=1,epochs=5,batch_size=128)
model.save('xception_model.h5')

score=model.evaluate(X_test,Y_test)
print('Accuracy is: '+ str(score))
