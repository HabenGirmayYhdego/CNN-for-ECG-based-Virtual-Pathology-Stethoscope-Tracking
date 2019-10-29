

# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 13:21:05 2019

@author: Haben
"""

## ECG classfication
import os
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.callbacks import ModelCheckpoint
from biosppy.signals import ecg
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import pandas as pd
import scipy.io as sio
from os import listdir
from os.path import isfile, join
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv2D, MaxPooling2D, Flatten, LSTM, Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras import regularizers
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from keras import backend as K
K.set_image_dim_ordering('th')
from keras import optimizers, losses, activations, models
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from keras.layers import Dense, Input, Dropout, Convolution1D, MaxPool1D, GlobalMaxPool1D, GlobalAveragePooling1D, \
    concatenate
from sklearn.metrics import f1_score, accuracy_score
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.preprocessing import Normalizer

from sklearn import preprocessing

np.random.seed(7)
os.chdir('D:/Implementation/ECG/ECG_Implementation')
print(os.getcwd())
#plt.style.use('ggplot')
dataframe=pd.read_csv('combined_FeatMorphLabel.csv', header = None)

#print(dataframe.describe())

#df.plot.scatter()
## model
shuffled = dataframe.sample(frac=1, axis=0)

#shuffled.to_csv('newfile.csv')

shuffled_cols=shuffled.loc[:,0:13]
#print(shuffled_cols.head())
shiffled_label=shuffled.loc[:,14]

#dataset = shuffled.values

#dataset[:,-1] = pd.factorize(dataset[:,-1])[0].astype(np.uint16)


#X_old=dataset[:,:-1]
#Y=dataset[:,-1]

#encoder = LabelEncoder()
#encoder.fit(Y_old)
#encoded_Y = encoder.transform(Y_old)
####convert integers to dummy variables (i.e. one hot encoded)
#Y = np_utils.to_categorical(encoded_Y)

# normalize features

#dataset = shuffled.values #returns a numpy array

normalized=(shuffled_cols-shuffled_cols.mean())/shuffled_cols.std()
print(normalized.head())
normalized_cocat=pd.concat([normalized,shiffled_label],axis=1,sort=False)
#min_max_scaler = preprocessing.MinMaxScaler()
#x_scaled = min_max_scaler.fit_transform(dataset)
#normalized = pd.DataFrame(x_scaled)

#X_train=np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))


#(X, X_test, Y, Y_test)= train_test_split(X,Y, test_size=0.2, random_state=seed)

Y = np.array( normalized_cocat[14].values).astype(np.int8)
X= np.array(normalized_cocat[list(range(14))].values)[..., np.newaxis]



#scaler = MinMaxScaler(feature_range=(0, 1))
#X = scaler.fit_transform(X_old][:])
#Y_test = np.array(df_test[187].values).astype(np.int8)
#X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]


(X, X_test, Y, Y_test)= train_test_split(X,Y, test_size=0.2, random_state=7)







def get_model0():
    nclass = 4
    inp = Input(shape=(14,1))
    img_1 = Convolution1D(256, kernel_size=5, activation=activations.relu, padding='same')(inp)
    img_1 = Convolution1D(256, kernel_size=5, activation=activations.relu, padding="same")(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.5)(img_1)
#    img_1 = Convolution1D(128, kernel_size=3, activation=activations.relu, padding="same")(img_1)
#    img_1 = Convolution1D(128, kernel_size=3, activation=activations.relu, padding="same")(img_1)
#    img_1 = MaxPool1D(pool_size=2)(img_1)
#    img_1 = Dropout(rate=0.4)(img_1)
#    img_1 = Convolution1D(256, kernel_size=5, activation=activations.relu, padding="valid")(img_1)
#    img_1 = Convolution1D(256, kernel_size=5, activation=activations.relu, padding="same")(img_1)
#    img_1 = MaxPool1D(pool_size=2)(img_1)
#    img_1 = Dropout(rate=0.4)(img_1)
    img_1 = Convolution1D(256, kernel_size=3, activation=activations.relu, padding="same")(img_1)
    img_1 = Convolution1D(256, kernel_size=3, activation=activations.relu, padding="same")(img_1)
    img_1 = GlobalMaxPool1D()(img_1)
    img_1 = Dropout(rate=0.4)(img_1)

    dense_1 = Dense(256, activation=activations.relu, name="dense_1")(img_1)
    dense_1 = Dense(256, activation=activations.relu, name="dense_2")(dense_1)
    dense_1 = Dense(nclass, activation=activations.softmax, name="dense_3_ecg")(dense_1)

    model = models.Model(inputs=inp, outputs=dense_1)
    opt = optimizers.Adam(0.001)

    model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
    model.summary()
    return model

def get_model1():
    nclass = 4
    inp = Input(shape=(14,1))
    img_1 = Convolution1D(8, kernel_size=5, activation=activations.relu, padding='same')(inp)
#    img_1 = Convolution1D(8, kernel_size=5, activation=activations.relu, padding="same")(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
#    img_1 = Convolution1D(16, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
#    img_1 = Convolution1D(16, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
#    img_1 = MaxPool1D(pool_size=2)(img_1)
#    img_1 = Dropout(rate=0.1)(img_1)
#    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
#    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
#    img_1 = MaxPool1D(pool_size=2)(img_1)
#    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(64, kernel_size=3, activation=activations.relu, padding="same")(img_1)
#    img_1 = Convolution1D(64, kernel_size=3, activation=activations.relu, padding="same")(img_1)
    img_1 = GlobalMaxPool1D()(img_1)
    img_1 = Dropout(rate=0.2)(img_1)

    dense_1 = Dense(64, activation=activations.relu, name="dense_1")(img_1)
    dense_1 = Dense(64, activation=activations.relu, name="dense_2")(dense_1)
    dense_1 = Dense(nclass, activation=activations.softmax, name="dense_3_mitbih")(dense_1)

    model = models.Model(inputs=inp, outputs=dense_1)
    opt = optimizers.Adam(0.001)

    model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
    model.summary()
    return model


def get_model2():
    nclass = 4
    inp = Input(shape=(14,1))
    img_1 = Convolution1D(8, kernel_size=5, activation=activations.relu, padding='same')(inp)
    img_1 = Convolution1D(8, kernel_size=5, activation=activations.relu, padding="same")(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(16, kernel_size=3, activation=activations.relu, padding="same")(img_1)
    img_1 = Convolution1D(16, kernel_size=3, activation=activations.relu, padding="same")(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="same")(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="same")(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(64, kernel_size=3, activation=activations.relu, padding="same")(img_1)
    img_1 = Convolution1D(64, kernel_size=3, activation=activations.relu, padding="same")(img_1)
    img_1 = GlobalMaxPool1D()(img_1)
    img_1 = Dropout(rate=0.2)(img_1)

    dense_1 = Dense(64, activation=activations.relu, name="dense_1")(img_1)
    dense_1 = Dense(64, activation=activations.relu, name="dense_2")(dense_1)
    dense_1 = Dense(nclass, activation=activations.softmax, name="dense_3_ecg")(dense_1)

    model = models.Model(inputs=inp, outputs=dense_1)
    opt = optimizers.Adam(0.1)

    model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
    model.summary()
    return model


smaller_model = get_model0()
#baseline_model = get_model1()
#bigger_model = get_model2()
file_path = "baseline_cnn_ecg.h5"
checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_acc", mode="max", patience=5, verbose=1)
redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=3, verbose=2)
callbacks_list = [checkpoint, early, redonplat]  # early

smaller_history=smaller_model.fit(X, Y, epochs=1000, verbose=2, callbacks=callbacks_list, validation_split=0.1)
#smaller_model.load_weights(file_path)

#baseline_history=baseline_model.fit(X, Y, epochs=1000, verbose=2, callbacks=callbacks_list, validation_split=0.1)
##baseline_model.load_weights(file_path)
#
#history=bigger_model.fit(X, Y, epochs=1000, verbose=2, callbacks=callbacks_list, validation_split=0.1)
##bigger_model.load_weights(file_path)


print(smaller_history.history.keys())
# summarize history for accuracy
plt.plot(smaller_history.history['acc'])
plt.plot(smaller_history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(smaller_history.history['loss'])
plt.plot(smaller_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


pred_test = smaller_model.predict(X_test)
pred_test = np.argmax(pred_test, axis=-1)

f1 = f1_score(Y_test, pred_test, average="macro")

print("Test f1 score : %s "% f1)

acc = accuracy_score(Y_test, pred_test)

print("Test accuracy score : %s "% acc)









results = confusion_matrix(Y_test, pred_test) 
print ('Confusion Matrix :')
print(results) 
print ('Accuracy Score :'),accuracy_score(Y_test, pred_test) 
print('Report :')
print (classification_report(Y_test, pred_test) )



