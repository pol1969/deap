# coding: utf-8
import pandas as pd
df = pd.read_csv('lk_1.csv',usecols=[0,5])
df['NO']=''
import numpy as np
np.random.randint(1,32,size=7)
np.random.choice(np.arange(1,32),7)
np.random.choice(np.arange(1,32),7)
np.random.choice(np.arange(1,32),7)
df.loc[38,'NO']=567
df
tmp = np.random.choice(np.arange(1,32),7).tobytes()
tmp
df.loc[df['ID']==64,'NO']=tmp
df.loc[df['ID']==64,'NO']
a = np.array([1,2,3,4,5])
df.loc[df['ID']==64,'NO']={"a":[a]}
df
df.loc[df['ID']==64,'NO'].values[0]['a']
df.loc[df['ID']==64,'NO'].values[0]['a'][0]
df.loc[df['ID']==64,'NO'].values[0]['a'][0][3]
