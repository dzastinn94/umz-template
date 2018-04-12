#!/usr/bin/python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



import numpy as np

from sklearn import linear_model

import os

os.chdir('/home/students/s407531/Desktop/uczeniem/umz-template/zajecia1/zadanie2/train')

r.pd.read_csv('train.tsv', sep='\t', names=['price', 'inNew', 'rooms', 'floor', 'location', 'sqr','sqrmeters'])


r=pd.read_csv('train.tsv', sep='\t', names=['price', 'inNew', 'rooms', 'floor', 'location', 'sqrmeters'])

reg=linear_model.LinearRegression()

reg.fit(pd.DataFrame(r, columns=['rooms']), r['price'])
#->out LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)


In [12]: reg.fit(pd.DataFrame(r, columns=['sqrmeters']), r['price'])
#-> out LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)


 os.chdir('/home/students/s407531/Desktop/uczeniem/umz-template/zajecia1/zadanie2/dev-0')

r=pd.read_csv('in.tsv', sep='\t', names=['inNew', 'rooms', 'floor', 'location', 'sqrmeters'])

r=pd.read_csv('train.tsv', sep='\t', names=['price', 'inNew', 'rooms', 'floor', 'location', 'sqrmeters'])


os.chdir('/home/students/s407531/Desktop/uczeniem/umz-template/zajecia1/zadanie2/train')

r=pd.read_csv('train.tsv', sep='\t', names=['price', 'inNew', 'rooms', 'floor', 'location', 'sqrmeters'])

os.chdir('/home/students/s407531/Desktop/uczeniem/umz-template/zajecia1/zadanie2/dev-0')

xyz=pd.read_csv('in.tsv', sep='\t', names=['inNew', 'rooms', 'floor', 'location', 'sqrmeters'])

y=xyz['sqrmeters']

y=y.values.reshape(-1,1)

y_p=reg.predict(y)

 y=xyz['sqrmeters'] 

 y=y.values.reshape(-1,1)

 y_p=reg.predict(y)

 pd.DataFrame(y_p).to_csv('out.tsv', sep='\t', index=False, header=False)

 os.chdir('/home/students/s407531/Desktop/uczeniem/umz-template/zajecia1/zadanie2/test-A')

 testA=pd.read_csv('in.tsv', sep='\t', names=['inNew', 'rooms', 'floor', 'location', 'sqrmeters'])

 zmienna=testA['sqrmeters']


testA.shape
#Out[33]: (150, 5)

zmienna=zmienna.values.reshape(-1,1)

y_zmienna=reg.predict(zmienna)


 pd.DataFrame(y_zmienna).to_csv('out.tsv', sep='\t', index=False, header=False)

sns.regplot(y=r['price'],x=r['sqrmeters']) 
#Out[42]: <matplotlib.axes._subplots.AxesSubplot at 0x7f9652f93b00>

plt.show()



