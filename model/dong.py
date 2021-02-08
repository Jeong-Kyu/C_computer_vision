import numpy as np
import pandas as pd
import tensorflow.keras.backend as K

# for i in range(1,6) :
#     df1.add(globals()['df{}'.format(i)], axis=1)
# df = df1.iloc[:,1:]
# df_2 = df1.iloc[:,:1]
# df_3 = (df/5).round(2)
# df_3.insert(0,'id',df_2)
# df3.to_csv('../data/csv/0122_timeseries_scale10.csv', index = False)
df1 = pd.read_csv(f'./0204_4_result.csv', index_col=0, header=0)
data1 = df1.to_numpy()

x = []
for i in range(2,5):
    df = pd.read_csv(f'./0204_{i}_result.csv', index_col=0, header=0)
    data = df.to_numpy()
    x.append(data)

x = np.array(x)
print(x.shape)
frequency={}
from collections import Counter
# df = pd.read_csv(f'c:/photovoltaics/csv/sample_submission2101{i}.csv', index_col=0, header=0)
for i in range(20480):
    for j in range(1):
        a = []
        for k in range(3):
            a.append(x[k,i,j].astype('int32'))
        # a = np.array(a)
        mode = Counter(a).most_common(2)
        r = mode[0][0]
        t = mode[0][1]
         
        if t!=1:
            df.iloc[[i],[j]] = r
        else:
            df.iloc[[i],[j]] = data1[i][j]
        # print(df)
        # for word in df.iloc[[i],[j]]:
        #     count = frequency.get(word,0)
        #     frequency[word] = count + 1

print(df)
        
y = pd.DataFrame(df, index = None, columns = None)
y.to_csv('./csv/submission210205_F.csv')