import pandas as pd
import numpy as np
import os
import time

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_hub as hub

def make_dataset(path, n_samples):
    df = pd.read_csv(path, usecols=[6,9], nrows=n_samples)
    df.columns = ['ratings', 'title']

    text = df['title'].tolist()
    text = [str(t).encode('ascii', 'replace') for t in text]
    text = np.array(text, dtype='object')

    labels = df['ratings'].tolist()
    labels = [1 if i>= 4 else 0 if i == 3 else -1 for i in labels]

    labels = np.array(pd.get_dummies(labels), dtype=int)

    return labels, text


def get_model():
    embed = hub.KerasLayer("https://tfhub.dev/google/nnlm-en-dim50/1",
     output_shape=[50],
     input_shape=[],
     dtype=tf.string,
     name='input',
     trainable=False)

    model = tf.keras.Sequential()
    model.add(embed)
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(3, activation='softmax', name='output'))
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['acc'])
    model.summary()
    return model

def train(epochs=5, bs=32):
    WORKDIR = os.getcwd()
    y_train, x_train = make_dataset('reviews_train.csv', n_samples=100000)
    y_val, x_val = make_dataset('reviews_test.csv', n_samples=10000)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                                filepath=os.path.join(WORKDIR,'model_checkpoint.h5'),
                                save_weights_only=False,
                                monitor='val_acc',
                                mode='auto',
                                save_best_only=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint('model-{epoch:03d}-{acc:03f}-{val_acc:03f}.h5', verbose=1, monitor='val_loss',save_best_only=True, mode='auto')
    model = get_model()
    model.fit(x_train, y_train, batch_size=bs, epochs=epochs, verbose=1,
         validation_data=(x_val, y_val), callbacks=[checkpoint])



if __name__ == "__main__":
    train()