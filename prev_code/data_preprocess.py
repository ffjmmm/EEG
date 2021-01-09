import utils
import pandas as pd

data = utils.MyDataNumpy(root='./dataset')
l = 0
r = 0
for i in range(len(data.label_test)):
    if data.label_test[i][0] > 0:
        l += 1
    if data.label_test[i][1] > 0:
        r += 1
print(l)
print(r)
print(data.label_test.shape)

data_train_dict = {}
for i in range(256):
    data_train_dict[i] = []
for i in range(len(data.data_train)):
    for j in range(256):
        data_train_dict[j].append(data.data_train[i][j])
dataframe = pd.DataFrame(data_train_dict)
dataframe.to_csv('data_train.csv', index=False)

label_train_dict = {}
label_train_dict['L'] = []
label_train_dict['R'] = []
for i in range(len(data.label_train)):
    label_train_dict['L'].append(data.label_train[i][0])
    label_train_dict['R'].append(data.label_train[i][1])
dataframe = pd.DataFrame(label_train_dict)
dataframe.to_csv('label_train.csv', index=False)

data_test_dict = {}
for i in range(256):
    data_test_dict[i] = []
for i in range(len(data.data_test)):
    for j in range(256):
        data_test_dict[j].append(data.data_test[i][j])
dataframe = pd.DataFrame(data_test_dict)
dataframe.to_csv('data_test.csv', index=False)

label_test_dict = {}
label_test_dict['L'] = []
label_test_dict['R'] = []
for i in range(len(data.label_test)):
    label_test_dict['L'].append(data.label_test[i][0])
    label_test_dict['R'].append(data.label_test[i][1])
dataframe = pd.DataFrame(label_test_dict)
dataframe.to_csv('label_test.csv', index=False)

