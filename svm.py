import csv
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from sklearn import svm
import pickle
from sklearn.metrics import confusion_matrix
from joblib import dump, load

f = open(r"./[DATA_INFORMATION].csv", 'rt', encoding='UTF8')
r = csv.reader(f)

headers = next(r)

print(headers)

label = []
data = []
ng_label = []
ok_label = []
ng_data = []
ok_data = []
count = 0
for row in r:
    tmp = []

    tmp.append(row[2])
    tmp.append(row[3])
    tmp.append(row[4])
    tmp.append(row[5])
    tmp.append(row[6])
    tmp.append(row[7])
    # tmp.append(row[7])

    if row[10] == "OK":
        ok_label.append("OK")
        ok_data.append(tmp)
    else:
        ng_label.append("NG")
        ng_data.append(tmp)

ok_index = np.random.permutation(len(ok_data))
ng_index = np.random.permutation(len(ng_data))

tmp_ng_data = []
tmp_ng_label = []
for i in ng_index:
    tmp_ng_data.append(ng_data[i])
    tmp_ng_label.append(ng_label[i])

tmp_ok_data = []
tmp_ok_label = []
for i in ok_index:
    tmp_ok_data.append(ok_data[i])
    tmp_ok_label.append(ok_label[i])

ok_part = int(len(tmp_ng_data)*0.8)
ng_part = int(len(tmp_ng_data)*0.8)
data = np.concatenate([tmp_ok_data[:ok_part], tmp_ng_data[:ng_part]])
label = np.concatenate([tmp_ok_label[:ok_part], tmp_ng_label[:ng_part]])

test_data = np.concatenate([tmp_ok_data[ok_part:], tmp_ng_data[ng_part:]])
test_label = np.concatenate([tmp_ok_label[ok_part:], tmp_ng_label[ng_part:]])

clf = svm.SVC(kernel='poly')
clf.fit(data,label)

dump(clf, "svm_model.joblib")


y_pred = clf.predict(test_data)
print(y_pred)
confusion = confusion_matrix(test_label, y_pred)
print(confusion)


