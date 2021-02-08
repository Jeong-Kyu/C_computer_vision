import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")

train = pd.read_csv('./csv/train.csv')
test = pd.read_csv('./csv/test.csv')
x_pred = test.drop(['id', 'letter'], axis=1).values
x = train.drop(['id', 'digit', 'letter'], axis=1).values
y = train['digit']

from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)
scaler.transform(x)
scaler.transform(x_pred)
x = x.reshape(-1, 28, 28, 1)
x_pred = x_pred.reshape(-1, 28, 28, 1)

# x = x_train/255

from keras.preprocessing.image import ImageDataGenerator
from sklearn.decomposition import PCA

idg = ImageDataGenerator(height_shift_range=(-1,1),width_shift_range=(-1,1)) # 이미지 카테고리화(4차원만 가능)
idg2 = ImageDataGenerator()

# pca = PCA()
# pca.fit(x)
# cumsum = np.cumsum(pca.explained_variance_ratio_)
# print('cumsum : ', cumsum)

# d = np.argmax(cumsum>0.98)+1
# print('cumsum >= 0.98', cumsum>=0.98)
# print('d : ', d) # 205

# pca =PCA(n_components = 784)
# pca.fit(x)
# x = pca.transform(x)
# x_pred = pca.transform(x_pred)
# from xgboost import XGBClassifier,plot_importance
# x = x.reshape(-1, 28, 28, 1)
# x_pred = x_pred.reshape(-1, 28, 28, 1)


# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 66 )


# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# encoder = OneHotEncoder()
# y_train = encoder.fit_transform(y_train.values.reshape(-1,1)).toarray()
# y_test = encoder.fit_transform(y_test.values.reshape(-1,1)).toarray()
# # y_train = np.zeros((len(y), len(y.unique())))
# # for i, digit in enumerate(y):
# #     y_train[i, digit] = 1


from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, Conv1D, MaxPool1D, LSTM, Input, BatchNormalization,MaxPooling2D, AveragePooling2D
def modeling():
    inputs1 = Input(shape=(28,28,1))
    bn = BatchNormalization(trainable=False)(inputs1)                                      # 배치정규화
    conv = Conv2D(512, kernel_size=5, strides=1, padding='same', activation='relu')(bn)   # CNN
    bn = BatchNormalization()(conv)                                                       # 배치정규화
    conv = Conv2D(512, kernel_size=2, strides=1, padding='same', activation='relu')(bn)   # CNN
    pool = AveragePooling2D()(conv)                                                     # Max Pooling

    bn = BatchNormalization()(pool)                                                       # 배치정규화
    conv = Conv2D(256, kernel_size=2, strides=1, padding='same', activation='relu')(bn)   # CNN
    bn = BatchNormalization()(conv)                                                       # 배치정규화
    conv = Conv2D(256, kernel_size=2, strides=1, padding='same', activation='relu')(bn)   # CNN
    pool = AveragePooling2D()(conv)                                                     # Max Pooling

    bn = BatchNormalization()(pool)                                                       # 배치정규화
    conv = Conv2D(128, kernel_size=2, strides=1, padding='same', activation='relu')(bn)   # CNN
    bn = BatchNormalization()(conv)                                                       # 배치정규화
    conv = Conv2D(128, kernel_size=2, strides=1, padding='same', activation='relu')(bn)   # CNN
    pool = AveragePooling2D()(conv) 

    flatten = Flatten()(pool)                                                             # Flatten
    bn = BatchNormalization()(flatten)                                                    # 배치정규화
    dense = Dense(1000, activation='relu')(bn)                                            # Fully Connected Layer
    bn = BatchNormalization()(dense)                                                      # 배치정규화
    outputs = Dense(10, activation='softmax')(bn)                                         # Fully Connected Layer

    model = Model(inputs=inputs1, outputs=outputs)
    return model

from sklearn.model_selection import StratifiedKFold
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau  
from keras.optimizers import Adam

re = ReduceLROnPlateau(patience=50, verbose=1, factor= 0.5)
ea = EarlyStopping(patience=100, verbose=1, mode='auto')
epochs = 1000
skf = StratifiedKFold(n_splits=15, random_state=42, shuffle=True) 
val_loss_min = []
result = 0
nth = 0

for train_index, valid_index in skf.split(x, y):
    x_train = x[train_index]
    x_valid = x[valid_index]
    y_train = y[train_index]
    y_valid = y[valid_index]
    # print(x_train.shape, x_valid.shape) #(1946, 28, 28, 1), (102, 28, 28, 1)
    # print(y_train.shape, y_valid.shape) #(1946,) (102,)

    # 실시간 데이터 증강을 사용해 배치에 대해서 모델을 학습(fit_generator에서 할 것)
    train_generator = idg.flow(x_train,y_train,batch_size=16) #훈련데이터셋을 제공할 제네레이터를 지정
    valid_generator = idg2.flow(x_valid,y_valid) # validation_data에 넣을 것
    test_generator = idg2.flow(x_pred,shuffle=False)  # predict(x_test)와 같은 역할
    

    model = modeling()
    mc = ModelCheckpoint('../data/modelcheckpoint/0204_2_best_mc.h5', save_best_only=True, verbose=1)
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer=Adam(lr=0.002,epsilon=None), metrics=['acc']) # y의 acc가 목적
    img_fit = model.fit_generator(train_generator,epochs=epochs, validation_data=valid_generator, callbacks=[ea,mc,re])
    
    # predict
    model.load_weights('../data/modelcheckpoint/0204_2_best_mc.h5')
    result += model.predict_generator(test_generator,verbose=True)/40 #a += b는 a= a+b
    # predict_generator 예측 결과는 클래스별 확률 벡터로 출력
    print('result:', result)

    # save val_loss
    hist = pd.DataFrame(img_fit.history)
    val_loss_min.append(hist['val_loss'].min())
    nth += 1
    print(nth, 'set complete!!') # n_splits 다 돌았는지 확인
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(x_train, y_train, epochs=500, validation_split=0.2)
# loss, acc = model.evaluate(x_test, y_test, batch_size=32)
# print(loss, acc)

submission = pd.read_csv('./csv/submission.csv')
submission['digit'] = np.argmax(model.predict(x_pred), axis=1)
submission.to_csv('210203 result2.csv', index=False)