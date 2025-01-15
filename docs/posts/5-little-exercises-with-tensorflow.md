---
slug: "5 little exercises with Tensorflow"
date: 2025-01-15
categories:
  - tensorflow
  - nlp
  - computer vision 
---

## Exercise 1 : Linear Regression

OBJECTIVE : Given this data, train a neural network to match the xs to the ys so that a predictor for a new value of X will give a float value very close to the desired answer, i.e. `print(model.predict([10.0]))` would give a satisfactory result.


```py

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


def solution_model():
    xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    ys = np.array([-4.0, 1.0, 6.0, 11.0, 16.0, 21.0], dtype=float)

    model = Sequential([
        layers.Input(shape=(1,)),
        layers.Dense(1)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.5),
        loss='mean_squared_error',
        metrics=['mean_squared_error']
    )
    history = model.fit(
        xs,
        ys,
        epochs=200
    )
    print(model.predict(np.array([10.])))
```

## Exercise 2 : Computer Vision (MNIST)

OBJECTIVE : Create and train a classifier for the MNIST dataset.

We expect it to classify 10 classes and the 
input shape should be the native size of the MNIST dataset which is 
28x28 monochrome. 

The input layer should accept (28,28) as the input shape only. 

```py
import tensorflow as tf

def solution_model():
    mnist = tf.keras.datasets.mnist
    input_shape = (28, 28)
    num_classes = 10

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # X_train has shape (N, 28, 28)
    # y_train has shape (N,)
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.expand_dims(inputs, axis=-1)
    x = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(64)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes)(x)

    model = tf.keras.Model(inputs, outputs)

    print(model.summary())

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1.e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    history = model.fit(
        x_train,
        y_train,
        epochs=10,
        validation_split=0.2
    )

    preds = model.predict(x_test)
```

## Exercise 3 : Computer Vision with CNNs

OBJECTIVE : Create and train a classifier to classify images between two classes
(damage and no_damage) using the satellite-images-of-hurricane-damage dataset.

The dataset can be found here:
https://ieee-dataport.org/open-access/detecting-damaged-buildings-post-hurricane-satellite-imagery-based-customized

The dataset consists of satellite images from Texas after Hurricane Harvey
divided into two groups (damage and no_damage).

We have already divided the data for training and validation.


We already know that:
1. The input shape of the model must be (128,128,3)
2. The last layer of the model must be a Dense layer with 1 neuron
  activated by sigmoid since this dataset has 2 classes.

We expect our neural network must have a validation accuracy of approximately 0.95 or above on the normalized validation dataset 


```py
import urllib
import zipfile

import tensorflow as tf

# This function downloads and extracts the dataset to the directory that
# contains this file.
def download_and_extract_data():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/certificate/satellitehurricaneimages.zip'
    urllib.request.urlretrieve(url, 'satellitehurricaneimages.zip')
    with zipfile.ZipFile('satellitehurricaneimages.zip', 'r') as zip_ref:
        zip_ref.extractall()

# This function normalizes the images.
def preprocess(image, label):
    image *= (1./255)
    return image, label


# This function loads the data, normalizes and resizes the images, splits it into
# train and validation sets, defines the model, compiles it and finally
# trains the model. The trained model is returned from this function.

def solution_model():
    # Downloads and extracts the dataset to the directory that
    # contains this file.
    download_and_extract_data()

    IMG_SIZE = (128, 128)
    BATCH_SIZE = 64

    # The following code reads the training and validation data from their
    # respective directories, resizes them into the specified image size
    # and splits them into batches. You must fill in the image_size
    # argument for both training and validation data.
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory='train/',
        image_size=IMG_SIZE
        , batch_size=BATCH_SIZE)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory='validation/',
        image_size=IMG_SIZE
        , batch_size=BATCH_SIZE)

    # Normalizes train and validation datasets using the
    # preprocess() function.
    # Also makes other calls, as evident from the code, to prepare them for
    # training.
    # Do not batch or resize the images in the dataset here since it's already
    # been done previously.
    train_ds = train_ds.map(
        preprocess,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    ).prefetch(tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.map(
        preprocess,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Code to define the model
    inputs = tf.keras.layers.Input(shape=(128, 128, 3))
    x = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(64)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(x)

    model = tf.keras.Model(inputs, outputs)

    print(model.summary())

    # Code to compile and train the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1.e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics='accuracy'
    )

    model.fit(
        train_ds,
        epochs=5,
        validation_data=val_ds
    )
```

## Exercise 4 : NLP

OBJECTIVE : Build and train a classifier for the sarcasm dataset.

The classifier should have a final layer with 1 neuron activated by sigmoid as shown.

The objective is to classify whether a sentence is sarcasm or not

```py
import json
import tensorflow as tf
import numpy as np
import urllib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def solution_model():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/sarcasm.json'
    urllib.request.urlretrieve(url, 'sarcasm.json')

    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_size = 20000

    with open('sarcasm.json') as f:
        data = json.load(f)

    sentences = [x['headline'] for x in data]
    labels = [x['is_sarcastic'] for x in data]

    print(len(sentences))

    train_sentences = sentences[:training_size]
    train_labels = labels[:training_size]
    val_sentences = sentences[training_size:]
    val_labels = labels[training_size:]

    # train_ds = tf.data.Dataset.from_tensor_slices((sentences, labels))
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=vocab_size,
        oov_token=oov_tok
    )
    tokenizer.fit_on_texts(train_sentences)

    train_sequences = tokenizer.texts_to_sequences(train_sentences)
    train_padded_seqs = tf.keras.preprocessing.sequence.pad_sequences(
        train_sequences,
        maxlen=max_length,
        padding=padding_type,
        truncating=trunc_type
    )

    val_sequences = tokenizer.texts_to_sequences(val_sentences)
    val_padded_seqs = tf.keras.preprocessing.sequence.pad_sequences(
        val_sequences,
        maxlen=max_length,
        padding=padding_type,
        truncating=trunc_type
    )

    train_ds = tf.data.Dataset.from_tensor_slices((train_padded_seqs, train_labels))
    eval_ds = tf.data.Dataset.from_tensor_slices((val_padded_seqs, val_labels))

    for x, y in train_ds.take(2):
        print(x)
        print(y)
        print(x.shape, y.shape)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(max_length,), dtype='int32'),
        tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=max_length,
            mask_zero=True),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    print(model.summary())

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5.e-4),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )

    history = model.fit(
        np.array(train_padded_seqs),
        np.array(train_labels),
        epochs=20,
        validation_data=(np.array(val_padded_seqs), np.array(val_labels)),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5),
            tf.keras.callbacks.ModelCheckpoint("./mymodel.h5", save_best_only=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5)
        ]
    )

```

## Exercise 5 : Time Series

OBJECTIVE : Build and train a neural network to predict time indexed variables of the multivariate house hold electric power consumption time series dataset. Using a window of past 24 observations of the 7 variables, the model should be trained to predict the next 24 observations of the 7 variables.

DAtaset can be found here :
https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption

The original Individual House Hold Electric Power Consumption Dataset
has Measurements of electric power consumption in one household with
a one-minute sampling rate over a period of almost 4 years.

Different electrical quantities and some sub-metering values are available.


Instructions:

1. Model input shape must be (BATCH_SIZE, N_PAST = 24, N_FEATURES = 7),
   since the testing infrastructure expects a window of past N_PAST = 24
   observations of the 7 features to predict the next N_FUTURE = 24
   observations of the same features.

2. Model output shape must be (BATCH_SIZE, N_FUTURE = 24, N_FEATURES = 7)

3. The last layer of your model must be a Dense layer with 7 neurons since
   the model is expected to predict observations of 7 features.


```py
import urllib
import zipfile

import pandas as pd
import tensorflow as tf


# This function downloads and extracts the dataset to the directory that
# contains this file.
def download_and_extract_data():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/certificate/household_power.zip'
    urllib.request.urlretrieve(url, 'household_power.zip')
    with zipfile.ZipFile('household_power.zip', 'r') as zip_ref:
        zip_ref.extractall()


# This function normalizes the dataset using min max scaling.
def normalize_series(data, min, max):
    data = data - min
    data = data / max
    return data


# This function is used to map the time series dataset into windows of
# features and respective targets, to prepare it for training and
# validation. First element of the first window will be the first element of
# the dataset. Consecutive windows are constructed by shifting
# the starting position of the first window forward, one at a time (indicated
# by shift=1). For a window of n_past number of observations of all the time
# indexed variables in the dataset, the target for the window
# is the next n_future number of observations of these variables, after the
# end of the window.

def windowed_dataset(series, batch_size, n_past=24, n_future=24, shift=1):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(size=n_past + n_future, shift=shift, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(n_past + n_future))
    ds = ds.map(lambda w: (w[:n_past], w[n_past:]))
    return ds.batch(batch_size).prefetch(1)



# This function loads the data from CSV file, normalizes the data and
# splits the dataset into train and validation data. It also uses
# windowed_dataset() to split the data into windows of observations and
# targets. Finally it defines, compiles and trains a neural network. This
# function returns the final trained model.

def solution_model():
    # Downloads and extracts the dataset to the directory that
    # contains this file.
    download_and_extract_data()
    # Reads the dataset from the CSV.
    df = pd.read_csv('household_power_consumption.csv', sep=',',
                     infer_datetime_format=True, index_col='datetime', header=0)

    # Number of features in the dataset. We use all features as predictors to
    # predict all features at future time steps.
    N_FEATURES = len(df.columns) # DO NOT CHANGE THIS

    # Normalizes the data
    data = df.values
    data = normalize_series(data, data.min(axis=0), data.max(axis=0))

    # Splits the data into training and validation sets.
    SPLIT_TIME = int(len(data) * 0.5) # DO NOT CHANGE THIS
    x_train = data[:SPLIT_TIME]
    x_valid = data[SPLIT_TIME:]

    tf.keras.backend.clear_session()
    tf.random.set_seed(42)

    BATCH_SIZE = 32

    # Number of past time steps based on which future observations should be
    # predicted
    N_PAST = 24 

    # Number of future time steps which are to be predicted.
    N_FUTURE = 24 

    # By how many positions the window slides to create a new window
    # of observations.
    SHIFT = 1 

    # Code to create windowed train and validation datasets.
    train_set = windowed_dataset(series=x_train, batch_size=BATCH_SIZE,
                                 n_past=N_PAST, n_future=N_FUTURE,
                                 shift=SHIFT)
    valid_set = windowed_dataset(series=x_valid, batch_size=BATCH_SIZE,
                                 n_past=N_PAST, n_future=N_FUTURE,
                                 shift=SHIFT)

    for x, y in train_set.take(1):
        print(x)
        print(y)

    # Code to define your model.
    inputs = tf.keras.layers.Input(shape=(N_PAST, N_FEATURES))
    x = tf.keras.layers.LSTM(64, return_sequences=True)(inputs)
    x = tf.keras.layers.Attention()([x, x])
    outputs = tf.keras.layers.Dense(N_FEATURES)(x)

    model = tf.keras.models.Model(inputs, outputs)

    # Code to train and compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=1.e-3)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.MAE,
        metrics='mean_average_error'
    )

    print(model.summary())

    model.fit(
        train_set,
        epochs=20,
        validation_data=valid_set,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5),
            tf.keras.callbacks.ModelCheckpoint("./mymodel.h5", save_best_only=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5)
        ]
    )

    return model
```

We can also test our model with :

```py
def mae(y_true, y_pred):
    return np.mean(abs(y_true.ravel() - y_pred.ravel()))


def model_forecast(model, series, window_size, batch_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(batch_size, drop_remainder=True).prefetch(1)
    forecast = model.predict(ds)
    return forecast

rnn_forecast = model_forecast(model, data, N_PAST, BATCH_SIZE)
rnn_forecast = rnn_forecast[SPLIT_TIME - N_PAST:-1, 0, :]

x_valid = x_valid[:rnn_forecast.shape[0]]
result = mae(x_valid, rnn_forecast)
```
