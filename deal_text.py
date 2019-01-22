import pickle
import numpy as np
import keras
from keras.layers import (Input, Activation, Conv1D, Conv2D, Dense, Dropout, Flatten, MaxPooling2D, MaxPooling1D)
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.models import Model

import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt

import os

def plot_history(history, result_dir):
    plt.plot(history.history['acc'], marker='.')
    plt.plot(history.history['val_acc'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.savefig(os.path.join(result_dir, 'model_accuracy.png'))
    plt.close()

    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(os.path.join(result_dir, 'model_loss.png'))
    plt.close()

def save_history(history, result_dir):
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    nb_epoch = len(acc)

    with open(os.path.join(result_dir, 'result.txt'), 'w') as fp:
        fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
        for i in range(nb_epoch):
            fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                i, loss[i], acc[i], val_loss[i], val_acc[i]))

text_w2v_path = './text_w2v_raw.pickle'

def loaddata(path):
    x = []
    y = []
    label = None
    with open('./labels.txt', 'r') as file:
        label = eval(file.read())
    with open(path, 'rb') as file:
        raw_data = pickle.load(file)
        for key in raw_data.keys():
            for i in range(len(label[key])):
                x.append(raw_data[key][i])
                y.append(label[key][i])

    np.savez('text_w2v_raw', x=x, y=y)

def main():
    lddt = np.load("text_w2v_raw.npz")
    X, Y = lddt['x'], lddt['y']
    Y = np_utils.to_categorical(Y, 3)

    ipt = Input(shape=(50, 300))

    c1d_1 = Conv1D(20, 3, activation='relu', border_mode='same')(Dropout(0.1)(ipt))
    c1d_2 = Conv1D(20, 4, activation='relu', border_mode='same')(Dropout(0.1)(ipt))
    lay_1 = keras.layers.concatenate([c1d_1, c1d_2], axis=-1)
    mp_1 = MaxPooling1D(2)(lay_1)
    c1d = Conv1D(40, 2, activation='relu')(Dropout(0.4)(mp_1))
    flt = Flatten()(c1d)
    den_1 = Dense(500, activation='relu')(Dropout(0.4)(flt))
    den_2 = Dense(100, activation='relu')(Dropout(0.4)(den_1))
    sftm = Dense(3, activation='softmax')(Dropout(0.4)(den_2))
    model = Model(inputs=ipt, outputs=sftm)
    model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['accuracy'])
    model.summary()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=43)
    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=100, epochs=100, verbose=1, shuffle=True)
    print(model.evaluate(X_test, Y_test, verbose=0))

    plot_history(history, './rstsss')
    save_history(history, './rstsss')

if __name__ == '__main__':
    main()