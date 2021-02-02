import numpy as np
import pandas as pd
# x_train = train.drop(['id','digit','letter'],1)
# x_train = x_train.values
train_data = pd.read_csv('./csv/train.csv',thousands = ',', encoding='UTF8', header=0)
# print(train_data.shape) (2048, 786)
submission = pd.read_csv('./csv/submission.csv')
x_pred =  pd.read_csv('./csv/test.csv',thousands = ',', encoding='UTF8', index_col=0, header=0)
x_pred = x_pred.iloc[:,1:]
x = train_data.drop(['id','digit','letter'],1)
x = x.values
x = x.reshape(-1,28,28,1)
y = train_data['digit']
y = y.values
from keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array
datagen = ImageDataGenerator(
                                 width_shift_range=5,
                                 height_shift_range=5,
                                 rotation_range=10,
                                 zoom_range=0.05) 

flow=datagen.flow(x,y,batch_size=1000,seed=2020)
x1,y1=flow.next()
print("x1.shape={}".format(x1.shape))
print("y1.shape={}".format(y1.shape))
x=x.reshape(2048,784)

x1=x1.reshape(1000,784)
y=y.reshape(2048,1)
y1=y1.reshape(1000,1)
x=np.append(x,x1,axis = 0)
y=np.append(y,y1,axis = 0)


# x_train = x_train.values
# print(x) #(2048, 784)
# print(y)# (2048,)
from keras.preprocessing.image import ImageDataGenerator
from sklearn.decomposition import PCA

# pca = PCA()
# pca.fit(x)
# cumsum = np.cumsum(pca.explained_variance_ratio_)
# print('cumsum : ', cumsum)

# d = np.argmax(cumsum>0.98)+1
# print('cumsum >= 0.98', cumsum>=0.98)
# print('d : ', d) # 205

pca =PCA(n_components = 205)
pca.fit(x)
x = pca.transform(x)
x_pred = pca.transform(x_pred)
from xgboost import XGBClassifier,plot_importance

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 66 )

model = XGBClassifier(n_estimators = 100,n_jobs=8)
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(model.feature_importances_)
print('acc : ', acc)

# acc :  0.23658536585365852

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

'''
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
encoder = OneHotEncoder()
y_train = encoder.fit_transform(y_train.reshape(-1,1)).toarray()
y_test = encoder.fit_transform(y_test.reshape(-1,1)).toarray()
x_train = x_train.reshape(2438,28,28,1)/255.0
x_test = x_test.reshape(610,28,28,1)/255.0
x_pred = x_pred.values.reshape(20480,28,28,1)/255.0

# print(x_train.shape) (1638, 98)
# print(x_test.shape) (410, 98)
# print(y_train.shape) (1638,)
# print(y_test.shape) (410,)

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, Conv1D, MaxPool1D, LSTM, Input, BatchNormalization,MaxPooling2D
inputs = Input(shape=(28,28,1))
bn = BatchNormalization(trainable=False)(inputs)                                      # 배치정규화
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

model = Model(inputs=inputs, outputs=outputs)

from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=20)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.5, verbose=1)
model.fit(x_train, y_train, validation_split=0.2, batch_size=128, epochs=200, callbacks=[es,reduce_lr])

loss, acc = model.evaluate(x_test, y_test, batch_size=128)
print('loss : ', loss)
print('acc : ', acc)


submission['digit'] = np.argmax(model.predict(x_pred), axis=1)
submission.to_csv('./csv/baseline.csv', index=False)
'''