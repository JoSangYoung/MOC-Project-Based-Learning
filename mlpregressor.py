import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers
import numpy as np
import csv
from datetime import datetime

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(16, activation='relu', input_shape=(6,)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(2, activation='sigmoid'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def main():
    f = open(r"./MOC_T1XX.csv", 'rt', encoding='UTF8')
    r = csv.reader(f)

    headers = next(r)

    print(headers)

    label = []
    data = []
    for row in r:
        tmp = []

        tmp.append(float(row[2]))
        tmp.append(float(row[3]))
        tmp.append(float(row[4]))
        tmp.append(float(row[5]))
        tmp.append(float(row[6]))
        tmp.append(float(row[7]))

        if(row[10] == "OK"):
            label.append(1)
        if(row[10] == "NG"):
            label.append(0)
        data.append(tmp)

    index = np.random.permutation(len(data))

    train_data = []
    train_label = []
    for i in index:
        train_data.append(data[i])
        train_label.append(label[i])

    train_label = tf.one_hot(train_label, 2, axis=-1)

    length = len(train_data)
    part = int(len(train_data)*0.8)
    
    data = train_data[:part]
    label = train_label[:part]

    test_data = train_data[part:]
    test_label = train_label[part:]

    index = np.random.permutation(len(data))
    train_data = []
    train_label = []
    for i in index:
        train_data.append(data[i])
        train_label.append(label[i])


    train_data = tf.convert_to_tensor(train_data)
    train_label = tf.convert_to_tensor(train_label)

    test_data = tf.convert_to_tensor(test_data)
    test_label = tf.convert_to_tensor(test_label)


    logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)


    print("build model")
    model = build_model()  # 새롭게 컴파일된 모델을 얻습니다.
    print("training model...")
    hist = model.fit(train_data, train_label, epochs=30, batch_size=16, verbose=0, validation_data=(test_data, test_label), callbacks=[tensorboard_callback])
    # print("evaluate model...")
    # loss, acc = model.evaluate(test_data, test_label)

    model.save("t1xx_model_2")

    # print("Test Loss",loss)
    # print("Test Accuracy",acc)


if __name__ == "__main__":
    main()