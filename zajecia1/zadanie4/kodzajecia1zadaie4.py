 #!/usr/bin/python3

 import seaborn as sns
 import numpy as np
 import matplotlib.pyplot as plt

 import pandas as pd

 from sklearn import linear_model

import os

 os.chdir('/home/students/s407531/Desktop/uczeniem/umz-template/zajecia1/zadanie4/train')

report=pd.read_csv('in.tsv', sep='\t', names=['price', 'mileage', 'year', 'brand', 'engingeType', 'engineCapacity'])

 r=pd.read_csv('in.tsv', sep='\t', names=['price', 'mileage', 'year', 'brand', 'engingeType', 'engineCapacity'])

 reg = linear_model.LinearRegression()

 replace_paliwo = {'gaz' : 2, 'diesel' : 3, 'benzyna' : 1}

 r=r.replace({'engingeType' : replace_paliwo})

reg.fit(pd.DataFrame(r, columns=['mileage', 'year', 'engineCapacity']), r['price'])
# LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)

x1=pd.DataFrame(r, columns=['mileage', 'year', 'engineCapacity'])

 y1=reg.predict(x1)

 pd.DataFrame(y1).to_csv('out.tsv', sep='\t', index=False, header=False)

 os.chdir('/home/students/s407531/Desktop/uczeniem/umz-template/zajecia1/zadanie4/dev-0')

 r2=pd.read_csv('in.tsv', sep='\t', names=['mileage', 'year', 'brand', 'engingeType', 'engineCapacity'])

r2=r2.replace({'engingeType' : replace_paliwo})

reg = linear_model.LinearRegression()

 reg.fit(pd.DataFrame(r, columns=['mileage', 'year', 'engineCapacity']), r['price'])
# LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)

x2=pd.DataFrame(r2, columns=['mileage', 'year', 'engineCapacity'])

 y2=reg.predict(x2)

 pd.DataFrame(y2).to_csv('out.tsv', sep='\t', index=False, header=False)

 os.chdir('/home/students/s407531/Desktop/uczeniem/umz-template/zajecia1/zadanie4/test-A')

 r3=pd.read_csv('in.tsv', sep='\t', names=['mileage', 'year', 'brand', 'engingeType', 'engineCapacity'])

 r3=r3.replace({'engingeType' : replace_paliwo})

 reg = linear_model.LinearRegression()

reg.fit(pd.DataFrame(r, columns=['mileage', 'year', 'engineCapacity']), r['price'])
# LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)

 x3=pd.DataFrame(r3, columns=['mileage', 'year', 'engineCapacity'])

 y3=reg.predict(x3)

pd.DataFrame(y3).to_csv('out.tsv', sep='\t', index=False, header=False)

 sns.regplot(y=r['price'], x=r['year'])
#<matplotlib.axes._subplots.AxesSubplot at 0x7f671c2e0828>

plt.show()

 

