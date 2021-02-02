import numpy as np
import pandas as pd

train_data = pd.read_csv('./csv/train.csv',thousands = ',', encoding='UTF8', index_col=0, header=0)
# print(train_data.shape) (2048, 786)
submission = pd.read_csv('./csv/submission.csv')
x_pred =  pd.read_csv('./csv/test.csv',thousands = ',', encoding='UTF8', index_col=0, header=0)
x_pred = x_pred.iloc[:,1:]
x = train_data.iloc[:,2:]
y = train_data.iloc[:,0]

# print(x.shape) (2048, 784)
# print(y.shape) (2048,)

from sklearn.decomposition import PCA

# pca = PCA()
# pca.fit(x)
# cumsum = np.cumsum(pca.explained_variance_ratio_)
# print('cumsum : ', cumsum)

# d = np.argmax(cumsum>0.98)+1
# print('cumsum >= 0.98', cumsum>=0.98)
# print('d : ', d) # 194

# pca =PCA(n_components = 194)
# pca.fit(x)
# x = pca.transform(x)
# x_pred = pca.transform(x_pred)
# from xgboost import XGBClassifier,plot_importance
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 66 )

# model = XGBClassifier(n_estimators = 100,n_jobs=8)
# model.fit(x_train, y_train)
# acc = model.score(x_test, y_test)
# print(model.feature_importances_)
# print('acc : ', acc)

# acc :  0.23658536585365852

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
encoder = OneHotEncoder()
y_train = encoder.fit_transform(y_train.values.reshape(-1,1)).toarray()
y_test = encoder.fit_transform(y_test.values.reshape(-1,1)).toarray()
x_train = x_train.reshape(1638,194)/255.0
x_test = x_test.reshape(410,194)/255.0
x_pred = x_pred.reshape(20480,194)/255.0
# print(x_train.shape) (1638, 98)
# print(x_test.shape) (410, 98)
# print(y_train.shape) (1638,)
# print(y_test.shape) (410,)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, Conv1D, MaxPool1D, LSTM

model = Sequential()
model.add(Dense(256,input_shape=(194,), activation='relu'))
model.add(Dense(128))
# model.add(Conv1D(filters=100, kernel_size=2,  input_shape=(194,1)))
# model.add(MaxPool1D(pool_size=2))
# model.add(LSTM(30,input_shape=(194,1), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(10, activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, batch_size=128, epochs=1000)

loss, acc = model.evaluate(x_test, y_test, batch_size=128)
print('loss : ', loss)
print('acc : ', acc)


submission['digit'] = np.argmax(model.predict(x_pred), axis=1)
submission.to_csv('./csv/baseline.csv', index=False)




'''
df_test = []

for i in range(81):
    file_path = './csv/test/' + str(i) + '.csv'
    temp = pd.read_csv(file_path)
    temp = preprocess_data(temp, is_train=False)
    df_test.append(temp)
# print(df_test)
# print('--------------------')
X_test = pd.concat(df_test)
X_test=X_test.to_numpy()
X_test=X_test.reshape(81,48,7)
# print(X_test) # (3888, 7)



from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x, y,train_size=0.8, random_state=0, shuffle=False)

print(x_train.shape)
print(x_val.shape)
# print(x1_train.shape)
# print(x1_val.shape)
print(y_train.shape)
# print(y1_train.shape)
print(y_val.shape)
# print(y1_val.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM,Conv1D, Dropout, Flatten
from tensorflow.keras.backend import mean, maximum

q_lst = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
def quantile_loss(i, q, y, pred):
  err = (y[i]-pred[i])
  return mean(maximum(q*err, (q-1)*err), axis=-1)

y0=[]
y9=[]
for q in q_lst:
    model = Sequential()
    model.add(Conv1D(filters = 100, kernel_size=1, input_shape=(48,7)))
    model.add(Dense(200))
    model.add(Dropout(0.4))
    model.add(Dense(200))
    model.add(Dropout(0.4))
    model.add(Dense(200))
    model.add(Dropout(0.4))
    model.add(Dense(200))
    model.add(Dropout(0.4))
    model.add(Dense(200))
    model.add(Dropout(0.4))
    model.add(Dense(200))
    model.add(Dropout(0.4))
    model.add(Dense(200))
    model.add(Dense(2))
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    es = EarlyStopping(monitor='val_loss', patience=20)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.5, verbose=1)
    
    model.compile(loss=lambda y,pred: quantile_loss(0, q,y,pred), optimizer='adam', metrics=[lambda y, pred: quantile_loss(0, q, y, pred)])
    hist1 = model.fit(x_train, y_train, batch_size=96, epochs=1, validation_split=0.2, callbacks=[es, reduce_lr])
    model.evaluate(x_val, y_val, batch_size=96)
    y1_pred = model.predict(X_test)
    y0.append(y1_pred)

    model.compile(loss=lambda y,pred: quantile_loss(1, q,y,pred), optimizer='adam', metrics=[lambda y, pred: quantile_loss(1, q, y, pred)])
    hist1 = model.fit(x_train, y_train, batch_size=96, epochs=1, validation_split=0.2, callbacks=[es, reduce_lr])
    model.evaluate(x_val, y_val, batch_size=96)
    y1_pred = model.predict(X_test)
    y9.append(y1_pred)

y0=np.array(y0)
Y0 = y0.transpose()
Y0 = Y0.reshape(7776,9)
# Y0 = pd.DataFrame(y0)
print(Y0.shape)

y9 =np.array(y9)
Y9 = y9.transpose()
Y9 = Y9.reshape(7776,9)
# Y9 = pd.DataFrame(y9)
print(Y9.shape)

Y5=[]
for w in range(81):
    Y5.append(Y0[(w*48):((w+1)*48),:])
    Y5.append(Y9[(w*48):((w+1)*48),:])

Y5 = np.asarray(Y5)

print(Y5)
print(Y5.shape)
Y5 = Y5.reshape(15552,9)
Y5 = pd.DataFrame(Y5)

index_c = []

for i in range(81):
    for a in range(2):
        for b in range(24):
            for c in range(2):
                index = str(i)+".csv_Day"+str(a+7)+"_"+str(b)+"h"+"%02d"%(30*(c))+"m"
                index_c.append(index)
Y5.columns = ['q_0.1','q_0.2','q_0.3','q_0.4','q_0.5','q_0.6','q_0.7','q_0.8','q_0.9']
Y5.index = index_c                
print(Y5)
Y5.to_csv('./csv/test1.csv', index=True)'''