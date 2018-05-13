
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

import pandas as pd
import os


lr_full = LogisticRegression()

r = pd.read_csv(os.path.join('train', 'train.tsv'), sep='\t', names =["Occupancy", "date", "Temperature", "Humidity", "Light", "CO2", "HumidityRatio"])
lr = LogisticRegression()

#lr.fit(r.CO2.values.reshape(-1, 1), r.Occupancy)


lr_full = LogisticRegression()

X = pd.DataFrame(r, columns=['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio'])

 #lr_full.fit(X, r.Occupancy)


#print('sensivity')

#print(tp/(tp+fn))
#0.9971081550028918

r = pd.read_csv(os.path.join('train', 'train.tsv'), sep='\t', names=[
                     "Occupancy", "date", "Temperature", "Humidity", "Light", "CO2", "HumidityRatio"])

print('TRAIN \n')

print(r.describe())

print('TRAIN \n')

print('Rozkład próby (%): ', end ='')
print(str(sum(r.Occupancy) / len(r)))

print('Zero rule :', end ='')
print(str(1 - sum(r.Occupancy) / len(r)))



lr = LogisticRegression()
X = pd.DataFrame(r, columns=['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio'])
lr.fit(X, r.Occupancy)




print('True Positives: ', end ='')
print(sum((lr.predict(r.CO2.values.reshape(-1, 1)) == r.Occupancy) & (lr.predict(r.CO2.values.reshape(-1, 1)) == 1)))
TP = sum((lr.predict(r.CO2.values.reshape(-1, 1)) == r.Occupancy) & (lr.predict(r.CO2.values.reshape(-1, 1)) == 1))
print('True Negatives: ', end ='')
print(sum((lr.predict(r.CO2.values.reshape(-1, 1)) == r.Occupancy) & (lr.predict(r.CO2.values.reshape(-1, 1)) == 0)))
TN = sum((lr.predict(r.CO2.values.reshape(-1, 1)) == r.Occupancy) & (lr.predict(r.CO2.values.reshape(-1, 1)) == 0))
print('False Positives: ', end ='')
print(sum((lr.predict(r.CO2.values.reshape(-1, 1)) != r.Occupancy) & (lr.predict(r.CO2.values.reshape(-1, 1)) == 1)))
FP = sum((lr.predict(r.CO2.values.reshape(-1, 1)) != r.Occupancy) & (lr.predict(r.CO2.values.reshape(-1, 1)) == 1))

print('False Negatives: ', end ='')
print(sum((lr.predict(r.CO2.values.reshape(-1, 1)) != r.Occupancy) & (lr.predict(r.CO2.values.reshape(-1, 1)) == 0)))
FN = sum((lr.predict(r.CO2.values.reshape(-1, 1)) != r.Occupancy) & (lr.predict(r.CO2.values.reshape(-1, 1)) == 0))





print('Dokładność: ', end ='')
print(str((TP + TN) / len(r)))

print('Czułość: ', end ='')
print(str(TP / (TP + FN)))

print('Swoistość: ', end ='')
print(str(TN / (FP + TN)))
print('\n')






rdev = pd.read_csv(os.path.join('dev-0', 'in.tsv'), sep='\t', names=["date", "Temperature", "Humidity", "Light", "CO2", "HumidityRatio"])
rdev = pd.DataFrame(rdev,columns = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"])
rdev_expected = pd.read_csv(os.path.join('dev-0', 'expected.tsv'), sep='\t', names=['y'])


print('DEV-0 \n')

print('Rozkład próby (%): ', end ='')
print(str(sum(rdev_expected['y']) / len(rdev_expected)))


print('zero rule: ', end ='')
print(str(1 - sum(rdev_expected['y']) / len(rdev)))




print('True Positives: ', end ='')
print(sum((lr.predict(rdev.CO2.values.reshape(-1, 1)) == rdev_expected['y']) & (lr.predict(rdev.CO2.values.reshape(-1, 1)) == 1)))
TP = sum((lr.predict(rdev.CO2.values.reshape(-1, 1)) == rdev_expected['y']) & (lr.predict(rdev.CO2.values.reshape(-1, 1)) == 1))

print('True Negatives: ', end ='')
print( sum((lr.predict(rdev.CO2.values.reshape(-1, 1)) == rdev_expected['y']) & (lr.predict(rdev.CO2.values.reshape(-1, 1)) == 0)))
TN = sum((lr.predict(rdev.CO2.values.reshape(-1, 1)) == rdev_expected['y']) & (lr.predict(rdev.CO2.values.reshape(-1, 1)) == 0))

print('False Positives: ', end ='')
print(sum((lr.predict(rdev) != rdev_expected['y']) & (lr.predict(rdev) == 'g')))
FP = sum((lr.predict(rdev) != rdev_expected['y']) & (lr.predict(rdev) == 'g'))

print('False Negatives: ', end ='')
print(sum((lr.predict(rdev.CO2.values.reshape(-1, 1)) != rdev_expected['y']) & (lr.predict(rdev.CO2.values.reshape(-1, 1)) == 0)))
FN = sum((lr.predict(rdev.CO2.values.reshape(-1, 1)) != rdev_expected['y']) & (lr.predict(rdev.CO2.values.reshape(-1, 1)) == 0))



print('Dokładność:  ', end ='')
print(str((TP + TN) / len(rdev)))

print('Czułość: ', end ='')
print(str(TP / (TP + FN)))

print('Swoistość: ', end ='')
print(str(TN / (FP + TN)))


rtest = pd.read_csv(os.path.join('test-A', 'in.tsv'), sep='\t', names=["date", "Temperature", "Humidity", "Light", "CO2", "HumidityRatio"])
rtest = pd.DataFrame(rtest,columns = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"])



file = open(os.path.join('dev-0', 'out.tsv'), 'w')

for line in list(lr.predict(rdev)):
   file.write(str(line)+'\n')





file = open(os.path.join('test-A', 'out.tsv'), 'w')

for line in list(lr.predict(rtest)):
   file.write(str(line) + '\n')




print('plotting...')



sns.regplot(x=rdev.CO2, y=rdev_expected.y, logistic=True, y_jitter=.1)
plt.savefig("wykres")




