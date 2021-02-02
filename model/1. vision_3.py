import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf

train = pd.read_csv('./csv/train.csv')
test = pd.read_csv('./csv/test.csv')

x_train = train.drop(['id', 'digit', 'letter'], axis=1).values
x_train = x_train.reshape(-1, 28, 28, 1)
x = x_train/255
y = train['digit']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 66 )

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
encoder = OneHotEncoder()
y_train = encoder.fit_transform(y_train.values.reshape(-1,1)).toarray()
y_test = encoder.fit_transform(y_test.values.reshape(-1,1)).toarray()
# y_train = np.zeros((len(y), len(y.unique())))
# for i, digit in enumerate(y):
#     y_train[i, digit] = 1


from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, Conv1D, MaxPool1D, LSTM, Input, BatchNormalization,MaxPooling2D
inputs1 = Input(shape=(28,28,1))
bn = BatchNormalization(trainable=False)(inputs1)                                      # 배치정규화
conv = Conv2D(128, kernel_size=5, strides=1, padding='same', activation='relu')(bn)   # CNN
bn = BatchNormalization()(conv)                                                       # 배치정규화
conv = Conv2D(128, kernel_size=2, strides=1, padding='same', activation='relu')(bn)   # CNN
pool = MaxPooling2D((2, 2))(conv)                                                     # Max Pooling

bn = BatchNormalization()(pool)                                                       # 배치정규화
conv = Conv2D(256, kernel_size=2, strides=1, padding='same', activation='relu')(bn)   # CNN
bn = BatchNormalization()(conv)                                                       # 배치정규화
conv = Conv2D(256, kernel_size=2, strides=1, padding='same', activation='relu')(bn)   # CNN
pool = MaxPooling2D((2, 2))(conv)                                                     # Max Pooling

bn = BatchNormalization()(pool)                                                       # 배치정규화
conv = Conv2D(512, kernel_size=2, strides=1, padding='same', activation='relu')(bn)   # CNN
bn = BatchNormalization()(conv)                                                       # 배치정규화
conv = Conv2D(512, kernel_size=2, strides=1, padding='same', activation='relu')(bn)   # CNN
pool = MaxPooling2D((2, 2))(conv) 


# inputs2 = Input(shape=(785,1))
# bn = BatchNormalization(trainable=False)(inputs2)                                      # 배치정규화
# conv = Conv1D(128, kernel_size=5, strides=1, padding='same', activation='relu')(bn)   # CNN
# bn = BatchNormalization()(conv)                                                       # 배치정규화
# pool = Conv1D(128, kernel_size=2, strides=1, padding='same', activation='relu')(bn)   # CNN
# flatten = Flatten()(pool)
# dense2 = Dense(128, activation='relu')(flatten)

# from tensorflow.keras.layers import concatenate, Concatenate
# merge1 = concatenate([dense1, dense2])

flatten = Flatten()(pool)                                                             # Flatten
bn = BatchNormalization()(flatten)                                                    # 배치정규화
dense = Dense(1000, activation='relu')(bn)                                            # Fully Connected Layer
bn = BatchNormalization()(dense)                                                      # 배치정규화
outputs = Dense(10, activation='softmax')(bn)                                         # Fully Connected Layer

model = Model(inputs=inputs1, outputs=outputs)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=200, validation_split=0.2,)
loss, acc = model.evaluate(x_test, y_test, batch_size=32)
print(loss, acc)
x_pred = test.drop(['id', 'letter'], axis=1).values
x_pred = x_pred.reshape(-1, 28, 28, 1)

submission = pd.read_csv('./csv/submission.csv')
submission['digit'] = np.argmax(model.predict(x_pred), axis=1)
submission.to_csv('baseline.csv', index=False)