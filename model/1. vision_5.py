import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf

train = pd.read_csv('./csv/train.csv')
test = pd.read_csv('./csv/test.csv')
x_pred = test.drop(['id', 'letter'], axis=1).values

x_train = train.drop(['id', 'digit', 'letter'], axis=1).values
# x_train = x_train.reshape(-1, 28, 28, 1)
x = x_train/255
y = train['digit']
from keras.preprocessing.image import ImageDataGenerator
from sklearn.decomposition import PCA

# pca = PCA()
# pca.fit(x)
# cumsum = np.cumsum(pca.explained_variance_ratio_)
# print('cumsum : ', cumsum)

# d = np.argmax(cumsum>0.98)+1
# print('cumsum >= 0.98', cumsum>=0.98)
# print('d : ', d) # 205

pca =PCA(n_components = 196)
pca.fit(x)
x = pca.transform(x)
x_pred = pca.transform(x_pred)
from xgboost import XGBClassifier,plot_importance
x = x.reshape(-1, 14, 14, 1)
x_pred = x_pred.reshape(-1, 14, 14, 1)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 66 )

# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test= train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 66 )

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
encoder = OneHotEncoder()
y_train = encoder.fit_transform(y_train.values.reshape(-1,1)).toarray()
y_test = encoder.fit_transform(y_test.values.reshape(-1,1)).toarray()
# y_train = np.zeros((len(y), len(y.unique())))
# for i, digit in enumerate(y):
#     y_train[i, digit] = 1


from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, Conv1D, MaxPool1D, LSTM, Input, BatchNormalization,MaxPooling2D
inputs1 = Input(shape=(14,14,1))
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

flatten = Flatten()(pool)                                                             # Flatten
bn = BatchNormalization()(flatten)                                                    # 배치정규화
dense = Dense(1000, activation='relu')(bn)                                            # Fully Connected Layer
bn = BatchNormalization()(dense)                                                      # 배치정규화
outputs = Dense(10, activation='softmax')(bn)                                         # Fully Connected Layer

model = Model(inputs=inputs1, outputs=outputs)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=500, validation_split=0.2)
loss, acc = model.evaluate(x_test, y_test, batch_size=32)
print(loss, acc)

submission = pd.read_csv('./csv/submission.csv')
submission['digit'] = np.argmax(model.predict(x_pred), axis=1)
submission.to_csv('210203 result.csv', index=False)