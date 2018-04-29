
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

import pandas as pd
import os


r = pd.read_csv(os.path.join('train', 'in.tsv'), sep='\t', header=None)


print('TRAIN \n')

print(r.describe())

print('TRAIN \n')

print('Rozkład próby (%): ', end ='')
print(str(sum(r[0] == 'g') / len(r)))

print('Zero rule :', end ='')
print(str(1 - sum(r[0] == 'g') / len(r)))



lr = LogisticRegression()
lr.fit(r[1].values.reshape(-1, 1), r[0])

print('True Positives: ', end ='')
print(sum((lr.predict(r[1].values.reshape(-1, 1)) == r[0]) & (lr.predict(r[1].values.reshape(-1, 1)) == 'g')))
TP = sum((lr.predict(r[1].values.reshape(-1, 1)) == r[0]) & (lr.predict(r[1].values.reshape(-1, 1)) == 'g'))

print('True Negatives: ', end ='')
print(sum((lr.predict(r[1].values.reshape(-1, 1)) == r[0]) & (lr.predict(r[1].values.reshape(-1, 1)) == 'b')))
TN = sum((lr.predict(r[1].values.reshape(-1, 1)) == r[0]) & (lr.predict(r[1].values.reshape(-1, 1)) == 'b'))

print('False Positives: ', end ='')
print(sum((lr.predict(r[1].values.reshape(-1, 1)) != r[0]) & (lr.predict(r[1].values.reshape(-1, 1)) == 'g')))
FP = sum((lr.predict(r[1].values.reshape(-1, 1)) != r[0]) & (lr.predict(r[1].values.reshape(-1, 1)) == 'g'))

print('False Negatives: ', end ='')
print(sum((lr.predict(r[1].values.reshape(-1, 1)) != r[0]) & (lr.predict(r[1].values.reshape(-1, 1)) == 'b')))
FN = sum((lr.predict(r[1].values.reshape(-1, 1)) != r[0]) & (lr.predict(r[1].values.reshape(-1, 1)) == 'b'))





print('Dokładność: ', end ='')
print(str((TP + TN) / len(r)))

print('Czułość: ', end ='')
print(str(TP / (TP + FN)))

print('Swoistość: ', end ='')
print(str(TN / (FP + TN)))
print('\n')






rdev = pd.read_csv(os.path.join('dev-0', 'in.tsv'), sep='\t', header=None)
rdev = pd.DataFrame(rdev)
rdev_expected = pd.read_csv(os.path.join('dev-0', 'expected.tsv'), sep='\t', names = ['y'])



print('DEV-0 \n')

print('Rozkład próby (%): ', end ='')
print(str(sum(rdev_expected['y'] == 'g') / len(rdev_expected)))


print('zero rule: ', end ='')
print(str(1 - sum(rdev_expected['y'] == 'g') / len(rdev)))


print('True Positives: ', end ='')
print(sum((lr.predict(rdev[0].values.reshape(-1, 1)) == rdev_expected['y']) & (lr.predict(rdev[0].values.reshape(-1, 1)) == 'g')))
TP = sum((lr.predict(rdev[0].values.reshape(-1, 1)) == rdev_expected['y']) & (lr.predict(rdev[0].values.reshape(-1, 1)) == 'g'))

print('True Negatives: ', end ='')
print(sum((lr.predict(rdev[0].values.reshape(-1, 1)) == rdev_expected['y']) & (lr.predict(rdev[0].values.reshape(-1, 1)) == 'b')))
TN = sum((lr.predict(rdev[0].values.reshape(-1, 1)) == rdev_expected['y']) & (lr.predict(rdev[0].values.reshape(-1, 1)) == 'b'))

print('False Positives: ', end ='')
print(sum((lr.predict(rdev[0].values.reshape(-1, 1)) != rdev_expected['y']) & (lr.predict(rdev[0].values.reshape(-1, 1)) == 'g')))
FP = sum((lr.predict(rdev[0].values.reshape(-1, 1)) != rdev_expected['y']) & (lr.predict(rdev[0].values.reshape(-1, 1)) == 'g'))

print('False Negatives: ', end ='')
print(sum((lr.predict(rdev[0].values.reshape(-1, 1)) != rdev_expected['y']) & (lr.predict(rdev[0].values.reshape(-1, 1)) == 'b')))
FN = sum((lr.predict(rdev[0].values.reshape(-1, 1)) != rdev_expected['y']) & (lr.predict(rdev[0].values.reshape(-1, 1)) == 'b'))



print('Dokładność:  ', end ='')
print(str((TP + TN) / len(rdev)))

print('Czułość: ', end ='')
print(str(TP / (TP + FN)))

print('Swoistość: ', end ='')
print(str(TN / (FP + TN)))



rtest = pd.read_csv(os.path.join('test-A', 'in.tsv'), sep='\t', header=None)
rtest = rtest[0].values.reshape(-1, 1)
rdev = rdev[0].values.reshape(-1, 1)



file = open(os.path.join('dev-0', 'out.tsv'), 'w')

for line in list(lr.predict(rdev)):
   file.write(str(line)+'\n')




file = open(os.path.join('test-A', 'out.tsv'), 'w')

for line in list(lr.predict(rtest)):
   file.write(str(line) + '\n')




print('plotting...')
#rdev_expected.[0]values.reshape(-1, 1) = rdev_expected..[0]values.reshape(-1, 1).map( x: 1 if x == "g" else 0)

#sns.regplot(x=rdev, y=rdev_expected.y, logistic=True, y_jitter=.1)
#plt.show()
