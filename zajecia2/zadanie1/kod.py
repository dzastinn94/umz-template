 import pandas as pd
 import seaborn as sns
    import os
    from sklearn.linear_model import LogisticRegression
    import matplotlib.pyplot as plt
    

 lr_full = LogisticRegression()

rtrain = pd.read_csv(os.path.join('train', 'train.tsv'), sep='\t', names
    =["Occupancy", "date", "Temperature", "Humidity", "Light", "CO2", "HumidityRatio"])

 lr = LogisticRegression()

lr.fit(rtrain.CO2.values.reshape(-1, 1), rtrain.Occupancy)


 lr_full = LogisticRegression()

 X = pd.DataFrame(rtrain, columns=['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio'])

lr_full.fit(X, rtrain.Occupancy)

tp = sum((lr_full.predict(X) == rtrain.Occupancy) & (lr_full.predict(X) == 1))

 fn = sum((lr_full.predict(X) != rtrain.Occupancy) & (lr_full.predict(X)  == 0))

 print('sensivity')

print(tp/(tp+fn))
#0.9971081550028918
