import pandas as pd
import seaborn as sns
import os
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


In [2]: lr_full = LogisticRegression()

rtrain = pd.read_csv(os.path.join('train', 'train.tsv'), sep='\t', names =["Occupancy", "date", "Temperature", "Humidity", "Light", "CO2", "HumidityRatio"])
lr = LogisticRegression()

 lr.fit(rtrain.CO2.values.reshape(-1, 1), rtrain.Occupancy)
Out[5]: LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)

lr_full = LogisticRegression()

X = pd.DataFrame(rtrain, columns=['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio'])

 lr_full.fit(X, rtrain.Occupancy)
Out[8]: LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)

tp = sum((lr_full.predict(X) == rtrain.Occupancy) & (lr_full.predict(X) == 1))

fn = sum((lr_full.predict(X) != rtrain.Occupancy) & (lr_full.predict(X) == 0))

print('sensivity')

print(tp/(tp+fn))
0.9971081550028918

