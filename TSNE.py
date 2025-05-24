import csv
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix
import seaborn as sns

f = open(r"./[DATA_INFORMATION].csv", 'rt', encoding='UTF8')
r = csv.reader(f)

headers = next(r)

print(headers)

# exit()

# data = pd.read_csv(r"./MOC_T1XX.csv")

ok_data = []
ok_label = []
ng_data = []
ng_label = []

for row in r:
    tmp = []

    tmp.append(row[2])
    tmp.append(row[3])
    tmp.append(row[4])
    tmp.append(row[5])
    tmp.append(row[6])
    tmp.append(row[7])

    if row[10] == "OK":
        ok_label.append("blue")
        ok_data.append(tmp)

    else:
        ng_label.append("red")
        ng_data.append(tmp)

part = 10000
data = np.concatenate([ng_data[:part], ok_data[:part]])
label = np.concatenate([ng_label[:part], ok_label[:part]])

print("ng_label num : ", len(ng_label))

model = TSNE(learning_rate=100)
transformed = model.fit_transform(data)

df_transformed = pd.DataFrame(data = transformed, columns=['Component_1','Component_2']).reset_index(drop=True)
df_transformed['Label'] = label
sns.scatterplot(data=df_transformed, x = 'Component_1',y='Component_2', hue='Label',s=5)
plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1))

# xs = transformed[:,0]
# ys = transformed[:,1]
# plt.scatter(xs,ys,c=label, s=1)
# plt.xlabel("t-SNE component 1")
# plt.ylabel("t-SNE component 2")


# plt.savefig('1TM_savefig_'+str(part)+'.png')




#     if not(row[8] == row[9]):
#         print(row[8])
#         print(row[9])
    

# model = TSNE(learning_rate=100)
# transformed = model.fit_transform(feature)

# xs = transformed[:,0]
# ys = transformed[:,1]
# plt.scatter(xs,ys,c=labels)

# plt.show()
