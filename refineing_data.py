import csv
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix
import pandas as pd

# data = pd.read_csv(r"./Data/MOC_P1_210929.csv")

data = pd.read_csv(r"./MOC_T1XX1.csv")



# for i, row in data.iterrows():
#     print(row["판정(L)"].where(data["판정(L)"] == "NG").count()))

# print(data["작업모델"].unique())

# TM_data = data.where(data["작업모델"] == "T1XX").dropna(how="all")

# print(data.where(data["판정(L)"] == "NG").count())

# TM_data.to_csv(r"./MOC_T1XX.csv")

# data.where(data["작업모델"] == "T1XX").dropna(how="all").to_csv(r"./MOC_T1XX1.csv")

print(data["판정(L)"].where(data["판정(L)"] == "NG").count())
print(data["판정(L)"].where(data["판정(L)"] == "OK").count())