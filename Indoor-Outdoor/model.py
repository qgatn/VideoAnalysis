import tensorflow as tf
import numpy as np
import pandas as pd
import keras
import os
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
import sklearn.model_selection as sk

# --- Declerations --- #
print("Entered Declerations........................")
IMAGE_H, IMAGE_W = 224, 224
NP_SAVE_2 = "extracted_data_2/"
W = ['lying', 'enclosed', 'male', 'speaking', 'sitting', 'unenclosed', 'walking', 'standing', 'female',
     'eating', 'anger', 'looking', 'fear', 'listening', 'touching', 'grief', 'love', 'running', 'telephoning',
     'joy', 'answering', 'arriving', 'slow', 'questioning', 'frenetic', 'departing', 'dense', 'affirming', 'negating',
     'farewelling', 'greeting', 'problem', 'musiconly']
w = ['enclosed','unenclosed']
count_w = {'enclosed':0,'unenclosed':0}
M_PATH = "models/"
df_details = pd.read_csv("dataframes/id_details_DF.csv")
df_label = pd.read_csv("dataframes/id_label_DF.csv")


def create_model():
    model = Sequential()

    # 1st layer group
    model.add(Convolution3D(64, (3, 3, 3), activation='relu',
                            data_format= "channels_last", name='conv1',
                            strides=(1, 1, 1), padding = 'same',
                            input_shape=(20, 224, 224, 3)))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                           padding='valid', name='pool1'))
    # 2nd layer group
    model.add(Convolution3D(128, (3, 3, 3), activation='relu',
                            padding='same', name='conv2',
                            strides=(1, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool2'))
    # 3rd layer group
    model.add(Convolution3D(256, (3, 3, 3), activation='relu',
                            padding='same', name='conv3a',
                            strides=(1, 1, 1)))
    model.add(Convolution3D(256, (3, 3, 3), activation='relu',
                            padding='same', name='conv3b',
                            strides=(1, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool3'))   

    model.add(Flatten())

    # FC layers group
    model.add(Dense(4096, activation='relu', name='fc6'))
    model.add(Dropout(.5))
    model.add(Dense(2048, activation='relu', name='fc7'))
    model.add(Dropout(.5))
    model.add(Dense(2, activation='sigmoid', name='sigmoid'))

    print(model.summary())

    return model


def get_data():
    X = []
    Y = []
    list = []

    for i in os.listdir(NP_SAVE_2):
        npy = i.find(".npy")
        list.append(int(i[:npy]))

    for i in list:
        chk = 0
        l_temp = []
        for j in w:
            l_temp.append(df_label[j][i])
            count_w[j] += df_label[j][i]
            chk+=df_label[j][i]

        if chk > 0:  # only accept data that has expression
            try:
                temp = np.load(NP_SAVE_2 + str(i) + ".npy")
            except:
                continue
            else:
                if temp.shape == (20, 224, 224, 3):
                    Y.append(l_temp)
                    X.append(temp)

    print(count_w)
    return np.asarray(X), np.asarray(Y)


# --- Main --- #
print("Getting data................")
X, Y = get_data()
#X = tf.cast(X, tf.float32)
#Y = tf.cast(Y, tf.int64)
print(X.shape, Y.shape)
x_train, x_test, y_train, y_test = sk.train_test_split(X,Y,test_size= 0.1,random_state=49)
x_train = tf.cast(x_train, tf.float32)
y_train = tf.cast(y_train, tf.int64)
x_test = tf.cast(x_test, tf.float32)
y_test = tf.cast(y_test, tf.int64)
print(x_train.shape, x_test.shape)
print(x_train.dtype, x_test.dtype)
print(y_train.shape, y_test.shape)
print(y_train.dtype, y_test.dtype)

epochs = 1
print("Starting Model.............")
name = M_PATH + "AV_enclosed_unenclosed_V1_1.h5"
model = create_model()
model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')
hist = model.fit(x = x_train, y = y_train, epochs = epochs, verbose = 1, callbacks = None, validation_split = 0.1, batch_size = 32)
model.save(name)

print("Evaluating Model:")
results = model.evaluate(x_test, y_test)
print(results)

print("Predicting")
predict = model.predict(x_test)

predict= predict.numpy()
y_test = y_test.numpy()
data = {'Target':y_test, 'Predict':predict}
final_df = pd.DataFrame(data)
final_df.to_csv((M_PATH+"AV_enclosed_unenclosed_V1_1_df.csv"), header=True)
print("All Completed")
