 #!/usr/bin/python3

import pandas as pd
import os
from sklearn import linear_model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

os.chdir('/home/justyna/Pulpit/uczeniem/umz-template/zajecia1/zadanie3/train')

r = pd.read_csv('train.tsv', sep = '\t',names = ['price', 'isNew','rooms', 'floor', 'location', 'sqrMetres'])

reg = linear_model.LinearRegression()

r.head()
r.corr()
sns.set(style='whitegrid', context='notebook')

c=['price', 'isNew', 'rooms', 'floor', 'sqrMetres']
sns.pairplot(r[c], size=3)
plt.show()

reg.fit(pd.dataFrame(r,columns=['sqrMetres', 'floor', 'rooms', 'isNew'], r['price'])

os.chdir('/home/justyna/Pulpit/uczeniem/umz-template/zajecia1/zadanie3/dev-0')
r2 = pd.read_csv('in.tsv', sep = '\t',names = ['isNew','rooms', 'floor', 'location', 'sqrMetres'])
x_d=pd.DataFrame(r2,columns=['sqrMetres', 'floor', 'rooms', 'isNew'])
y_d=reg.predict(x_d)
y_d=pd.Series(y_d)
y_d.to_csv('out.tsv', sep='\t', header=False, index=False)

os.chdir('/home/justyna/Pulpit/uczeniem/umz-template/zajecia1/zadanie3/test-A')

r3 = pd.read_csv('in.tsv', sep = '\t',names = ['isNew','rooms', 'floor', 'location', 'sqrMetres'])
x_d2=pd.DataFrame(r3,columns=['sqrMetres', 'floor', 'rooms', 'isNew'])
y_d2=reg.predict(x_d2)
y_d2=pd.Series(y_d2)
y_d2.to_csv('out.tsv', sep='\t', header=False, index=False)

sns.regplot(y=r['price'], x=r['rooms'])


