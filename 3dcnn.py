import argparse
import os

import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import cifar10
from keras.layers import (Activation, Conv3D, Dense, Dropout, Flatten,
                          MaxPooling3D)
from keras.layers.advanced_activations import LeakyReLU
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split

import video_prp


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


def loaddata(video_dir, label_path, vid3d, result_dir, color=False, skip=True):
    files = os.listdir(video_dir)
    X = []
    labels = []
    labellist = []

    for file in files:
        vdname = file[:file.rfind('_')]
        no = int(file[file.rfind('_')+1:file.rfind('.')])
        X.append(vid3d.video3d(video_dir+file))
        raw_label = None
        with open(label_path, 'r') as f:
            raw_label = eval(f.read())
        label = raw_label[vdname][no-1]
        if label < 0:
            label = 0
        elif label > 0:
            label = 2
        else:
            label = 1
        labels.append(label)

    np.savez(result_dir, x=X, y=labels)
    print('saved ')

def main(label_path, result_doc):
    lddt = np.load(label_path)
    X, Y = lddt['x'], lddt['y']
    Y = np_utils.to_categorical(Y, 3)

    # Define model
    model = Sequential()
    model.add(Conv3D(32, kernel_size=(5, 5, 5), input_shape=(
        X.shape[1:]), border_mode='same', activation='relu'))

    model.add(MaxPooling3D(pool_size=(3, 3, 3), border_mode='same'))

    model.add(Flatten())
    model.add(Dense(300, activation='relu'))
    model.add(Dense(100, activation='softmax'))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss=categorical_crossentropy,
                  optimizer=Adam(), metrics=['accuracy'])
    model.summary()
    plot_model(model, show_shapes=True,
               to_file=os.path.join(result_doc, 'model.png'))

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=43)

    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=100,
                        epochs=100, verbose=1, shuffle=True)
    model.evaluate(X_test, Y_test, verbose=0)
    model_json = model.to_json()
    with open(os.path.join(result_doc, 'ucf101_3dcnnmodel.json'), 'w') as json_file:
        json_file.write(model_json)
    model.save(os.path.join(result_doc, 'ucf101_3dcnnmodel.hd5'))

    loss, acc = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', loss)
    print('Test accuracy:', acc)
    plot_history(history, result_doc)
    save_history(history, result_doc)


if __name__ == '__main__':
    v3d = video_prp.Videoto3D()
    vddir = 'C:/Users/96225/Desktop/Segmented/'
    label_dir = r'./labels.txt'
    resultdir = r'./result/result.npz'
    loaddata(vddir, label_dir, v3d, resultdir)
    main(resultdir, './result/')