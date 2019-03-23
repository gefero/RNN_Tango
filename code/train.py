#!/usr/bin/python
# coding: utf-8

#%%
from __future__ import print_function
import numpy as np
import os
import random
import sys
#%%

# Cargamos las letras

path = '../data/V2letras.txt'
with open(path, encoding='utf-8') as f:
    text_orig = f.read().lower()

text = text_orig
#%%
from keras.callbacks import LambdaCallback, ModelCheckpoint, TensorBoard, EarlyStopping
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.optimizers import RMSprop
#%%
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1
#%%

#2, 128, 0.5, RMSprop(), 0.005, 'categorical_crossentropy', ['accuracy']

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("layers", help="n of layers in rnn",
                    type=int)
    parser.add_argument("neurons", help="n of neurons in each layer",
                    type=int)
    parser.add_argument("dropout_rate", help="drop out rate",
                    type=float)
    parser.add_argument("lrate", help="learning rate",
                    type=float)
    parser.add_argument("loss_metric", help="loss metric",
                    type=str)
    parser.add_argument("val_metric", help="Validation metric (list of strings)",
                    type=str)
    args = parser.parse_args()



print(args)
print('OK')

def build_rnn(layers=args.layers, neurons=args.neurons, 
              dropout_rate=args.dropout_rate, lrate=args.lrate, 
              loss_metric=args.loss_metric, 
              val_metric=args.val_metric, 
              in_shape=(maxlen, len(chars))):
    
    print('Building model...')
    model = Sequential()
    model.add(LSTM(neurons, input_shape=in_shape, return_sequences=True))
    
    for l in range(1, layers):
        
        model.add(Dropout(dropout_rate))
        model.add(LSTM(neurons))
        model.add(Dropout(dropout_rate))
    
    model.add(Dense(len(chars), activation='softmax'))

    optimizer = RMSprop(lr=lrate)
    model.compile(loss=loss_metric, metrics=[val_metric],
                      optimizer=optimizer)
    print(model.summary())
    return(model)
#%%

model = build_rnn()

#%%

# build callbacks

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, _):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(text) - maxlen - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()

print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

checkpoint_path = "../chkpt/lstm_2_128_M.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback
cp_callback = ModelCheckpoint(checkpoint_path, 
                              save_weights_only=False,
                              save_best_only=True, 
                              monitor='loss',
                              mode='min',
                              verbose=1)


from keras.callbacks import TensorBoard
tb_callback = TensorBoard(log_dir='./logs', 
                         histogram_freq=0,
                         write_graph=False,
                         write_grads=True,
                         batch_size=10,
                         update_freq='epoch',
                         write_images=True)


# Create EarlyStopping Callback

es_callback = EarlyStopping(monitor='loss', patience=2)

#%%

model.fit(x, y,
          batch_size=128,
          epochs=7,
          callbacks=[cp_callback, es_callback, print_callback])

#%%
model_path = '../model/lstm_2_128.hdf5'
model.save(model_path)