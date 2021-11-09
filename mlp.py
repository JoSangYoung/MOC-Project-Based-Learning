import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers
import numpy as np
import csv
from datetime import datetime
from tensorflow.math import confusion_matrix
import pandas as pd
import random

random.seed(42)
np.random.seed(42)

def build_model():
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)

    model = models.Sequential()
    model.add(layers.Dense(256, activation='relu', input_shape=(6,), name="layer1" ))
    model.add(layers.Dense(256, activation='relu', name="layer2"))
    model.add(layers.Dense(128, activation='relu', name="layer3"))
    model.add(layers.Dense(128, activation='relu', name="layer4"))
    model.add(layers.Dense(16, activation='relu', name="layer7"))
    model.add(layers.Dense(2, activation='softmax', name="layer8"))
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.FalsePositives(name='false_positives'), tf.keras.metrics.FalseNegatives(name='false_negatives')])

    return model

def Shuffle(data, label):

    index = np.random.permutation(len(label))

    out_data = []
    out_label = []
    for i in index:
        out_data.append(data[i])
        out_label.append(label[i])

    return out_data, out_label

def Get_DataFrame(filename, product_name):
    df = pd.read_csv(filename).drop(['Unnamed: 0'],axis=1)
    df = df[df.columns[[0,1,2,3,4,5,8,11,12,13,14]].tolist()]
    df['생산날짜(L)']=pd.to_datetime(df['생산날짜(L)'],format='%Y-%m-%d %H:%M:%S')
    df['Tester']=df['Tester'].astype('category')
    df['Machine']=df['Machine'].astype('category')
    df['판정(L)']=df['판정(L)'].astype('category')
    df['작업모델']=df['작업모델'].astype('category')
    df = df[df['작업모델']==product_name].reset_index(drop=True).drop('작업모델',axis=1)

    df_2021 = df[df['생산날짜(L)'] > pd.to_datetime('2021-1-1')].reset_index(drop=True)

    n_sample = int(df_2021[df_2021["판정(L)"] == "NG"]["판정(L)"].count())

    ok_df = df_2021[df_2021['판정(L)'] == 'OK'].sample(n_sample)
    ng_df = df_2021[df_2021['판정(L)'] == 'NG'].sample(n_sample)

    ok_df.drop(['생산날짜(L)','Machine','Tester'],axis=1,inplace=True)
    ng_df.drop(['생산날짜(L)','Machine','Tester'],axis=1,inplace=True)

    ok_df["판정(L)"] = 1
    ng_df["판정(L)"] = 0

    return ok_df, ng_df, n_sample

def Set_Dataset(ok_df, ng_df, n_sample):
    ok_data_samples = ok_df[ok_df.columns[[0,1,2,3,4,5]].tolist()].to_numpy()
    ok_label_samples = ok_df[ok_df.columns[[6]].tolist()].to_numpy()
    ok_label_samples = ok_label_samples.reshape(ok_label_samples.shape[0])

    ok_data_samples, ok_label_samples = Shuffle(ok_data_samples, ok_label_samples)

    ng_data_samples = ng_df[ng_df.columns[[0,1,2,3,4,5]].tolist()].to_numpy()
    ng_label_samples = ng_df[ng_df.columns[[6]].tolist()].to_numpy()
    ng_label_samples = ng_label_samples.reshape(ng_label_samples.shape[0])


    ng_data_samples, ng_label_samples = Shuffle(ng_data_samples, ng_label_samples)
    
    # Train / Test 나누기 (0.8 : 0.2 비율로 하였음)
    part = int(n_sample * 0.8)
    train_data = np.concatenate([ok_data_samples[:part], ng_data_samples[:part]])
    train_label = np.concatenate([ok_label_samples[:part], ng_label_samples[:part]])
    train_data, train_label = Shuffle(train_data,  train_label)

    test_data = np.concatenate([ok_data_samples[part:], ng_data_samples[part:]])
    test_label = np.concatenate([ok_label_samples[part:], ng_label_samples[part:]])
    test_data, test_label = Shuffle(test_data, test_label)

    # One hot encoding
    train_label = tf.one_hot(train_label, 2, axis=-1)
    test_label = tf.one_hot(test_label, 2, axis=-1)
    print(train_label.shape)
    print(test_label.shape)
    # Tensor 형태로 convert
    train_data = tf.convert_to_tensor(train_data)
    train_label = tf.convert_to_tensor(train_label)

    test_data = tf.convert_to_tensor(test_data)
    test_label = tf.convert_to_tensor(test_label)

    return train_data, train_label, test_data, test_label


def main():
    filename = "./MOC_P1_210929.csv"
    product_name = "T1XX"
    logdir = "logs/scalars/"+ product_name + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

    ok_df, ng_df, n_sample = Get_DataFrame(filename, product_name)
    print(ok_df.head(5))
    print(ng_df.head(5))

    train_data, train_label, test_data, test_label = Set_Dataset(ok_df, ng_df, n_sample)

    print("build model")
    model = build_model()  # 새롭게 컴파일된 모델을 얻습니다.

    print("training model...")
    hist = model.fit(train_data, train_label, epochs=1000, batch_size=32, verbose=0, callbacks=[tensorboard_callback])

    print(hist.history['accuracy'][-1])
    eva_result = model.evaluate(test_data, test_label)

    model.save("./model"+ product_name + datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    print("loss, acc, precision, recall, falsepositives, falsenegatives:", eva_result)



if __name__ == "__main__":
    main()