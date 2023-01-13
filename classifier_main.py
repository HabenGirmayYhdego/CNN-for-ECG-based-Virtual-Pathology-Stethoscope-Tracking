

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
from keras.layers import UpSampling1D
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense,BatchNormalization
from keras.layers import LSTM
from keras.models import Model
from sklearn.metrics import f1_score, accuracy_score
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.preprocessing import Normalizer
from keras.layers import Layer,MultiHeadAttention, LayerNormalization
from sklearn import preprocessing

np.random.seed(7)
os.chdir('D:/Implementation/ECG/ECG_Implementation')
print(os.getcwd())
#plt.style.use('ggplot')
# dataframe=pd.read_csv('combined_FeatMorphLabel.csv', header = None)
dataframe=pd.read_csv('combined_Feat.csv', header = None)
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







def MLP_MOdel():
    nclass = 4
    input_features = Input(shape=(14,1))
    # Define the model
    model = Sequential()
    model.add(Dense(64, input_dim=input_features, activation='relu'))
    model.add(Dropout(0.5))  # dropout rate of 0.5
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))  # dropout rate of 0.5
    model.add(Dense(nclass, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()
    return model

def CNN_Model():
    nclass = 4
    input_features = Input(shape=(14,1))
    # Define the first convolutional layer
    conv1 = Conv1D(filters=128, kernel_size=5, activation='relu')(input_features)
    conv1 = Conv1D(filters=128, kernel_size=5, activation='relu')(conv1)
    pool1 = MaxPooling1D(pool_size=2)(conv1)
    drop1=Dropout(0.5)(pool1)

    # Define the second convolutional layer
    conv2 = Conv1D(filters=128, kernel_size=5, activation='relu')(drop1)
    conv2 = Conv1D(filters=128, kernel_size=5, activation='relu')(conv2)
    pool2 = MaxPooling1D(pool_size=2)(conv2)
    drop2 = Dropout(0.5)(pool2)

    # Flatten the output of the second pooling layer
    flat = Flatten()(drop2)

    # Define the output layer
    output_layer = Dense(nclass, activation='softmax')(flat)

    # Create the model
    model = Model(inputs=input_features, outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()
    return model

def LSTM_MOdel():
    nclass = 4
    input_features = Input(shape=(14,1))
    # Define the first LSTM layer
    lstm1 = LSTM(units=128, return_sequences=True)(input_features)
    # Define the second LSTM layer
    lstm2 = LSTM(units=64)(lstm1)
    drop1 = Dropout(0.5)(lstm2)
    output_layer = Dense(64, activation='relu')(drop1)
    # Define the output layer
    output_layer = Dense(nclass, activation='softmax')(output_layer)

    # Create the model
    model = Model(inputs=input_features, outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()
    return model

def FCN_Model():
    nclass = 4
    input_features = Input(shape=(14,1))
    conv1 = Conv1D(filters=512, kernel_size=3, activation='relu')(input_features)
    bn1 = BatchNormalization()(conv1)
    pool1 = MaxPooling1D(pool_size=2)(bn1)

    # Define the second convolutional layer
    conv2 = Conv1D(filters=256, kernel_size=3, activation='relu')(pool1)
    bn2 = BatchNormalization()(conv2)
    pool2 = MaxPooling1D(pool_size=2)(bn2)

    conv3 = Conv1D(filters=128, kernel_size=3, activation='relu')(pool2)
    bn3 = BatchNormalization()(conv3)
    pool3 = MaxPooling1D(pool_size=2)(bn3)
    # Define the upsampling layer
    up = UpSampling1D(size=2)(pool3)

    # Define the output layer
    output_layer = Dense(nclass, activation='softmax')(up)

    # Create the model
    model = Model(inputs=input_features, outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()
    return model
def CNNLSTM_MOdel():
    nclass = 4
    input_features = Input(shape=(14,1))
    # Define the first convolutional layer
    conv1 = Conv1D(filters=256, kernel_size=3, activation='relu')(input_features)
    conv1 = Conv1D(filters=128, kernel_size=3, activation='relu')(conv1)
    pool1 = MaxPooling1D(pool_size=2)(conv1)
    drop1 = Dropout(0.5)(pool1)
    # Flatten the output of the second pooling layer
    flat = Flatten()(drop1)
    # Define the LSTM layer
    lstm = LSTM(units=32)(flat)
    # Define the output layer
    output_layer = Dense(64, activation='relu')(lstm)
    output_layer = Dense(nclass, activation='softmax')(output_layer)

    # Create the model
    model = Model(inputs=input_features, outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()
    return model
class TransformerBlock(Layer):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attn = MultiHeadAttention(d_model, nhead)
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.dropout = Dropout(dropout)

    def call(self, inputs):
        attn_output, _ = self.attn(inputs)
        x = self.dropout(attn_output)
        x = self.norm1(x + inputs)
        x = self.norm2(x)
        return x


def Transformer_MOdel():
    nclass = 4
    input_features = Input(shape=(14,1))
    # Add the Transformer block to the input layer
    x = TransformerBlock(d_model=128, nhead=8)(input_features)

    # Add a dense layer for classification
    output_layer = Dense(nclass, activation='softmax')(x)

    # Create the model
    model = Model(input_features, output_layer)
    model.compile(optimizer='adam', loss='binary_crossentropy')

    return model

model = CNN_Model()
#baseline_model = get_model1()
#bigger_model = get_model2()
file_path = "baseline_cnn_ecg.h5"
checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_acc", mode="max", patience=5, verbose=1)
redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=3, verbose=2)
callbacks_list = [checkpoint, early, redonplat]  # early

smaller_history=model.fit(X, Y, epochs=1000, verbose=2, callbacks=callbacks_list, validation_split=0.1)
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


pred_test = model.predict(X_test)
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



