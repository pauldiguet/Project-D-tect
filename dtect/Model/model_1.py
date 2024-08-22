from tensorflow import keras
from keras import Sequential, layers, models
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D, Input
from keras.models import Model
import numpy as np

input_shape = (544,544,3)

def build_encoder(input_shape):

    encoder = Sequential()

    encoder.add(layers.Conv2D(32, (3,3), input_shape=input_shape, padding='same', activation='relu'))
    encoder.add(layers.MaxPooling2D(2,2))

    encoder.add(layers.Conv2D(64, (3,3),padding='same', activation='relu'))
    encoder.add(layers.MaxPooling2D(2,2))

    encoder.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
    encoder.add(layers.MaxPooling2D(2,2))

    encoder.add(layers.Conv2D(256, (3,3), padding='same', activation='relu'))
    encoder.add(layers.MaxPooling2D(2,2))

    encoder.add(layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
    encoder.add(layers.MaxPooling2D(2, 2))

    return encoder


def build_decoder():

    decoder = Sequential()
    decoder.add(Conv2DTranspose(512, (3, 3), padding='same', activation='relu', input_shape=(17, 17, 512)))
    decoder.add(UpSampling2D((2, 2)))

    decoder.add(Conv2DTranspose(256, (3, 3), padding='same', activation='relu'))
    decoder.add(UpSampling2D((2, 2)))

    decoder.add(Conv2DTranspose(128, (3, 3), padding='same', activation='relu'))
    decoder.add(UpSampling2D((2, 2)))

    decoder.add(Conv2DTranspose(64, (3, 3), padding='same', activation='relu'))
    decoder.add(UpSampling2D((2, 2)))

    decoder.add(Conv2DTranspose(32, (3, 3), padding='same', activation='relu'))
    decoder.add(UpSampling2D((2, 2)))

    decoder.add(Conv2DTranspose(3, (3, 3), padding='same', activation='sigmoid'))

    return decoder


def build_autoencoder(encoder, decoder):
    inp = Input(input_shape)
    encoded = encoder(inp)
    decoded = decoder(encoded)
    autoencoder = Model(inp, decoded)
    return autoencoder


def compile_autoencoder(autoencoder):
    autoencoder.compile(loss='mean_squared_error',
                  optimizer='adam',
                    metrics=['accuracy'])


def train_autoencoder(X_train, y_train, input_shape):
    encoder = build_encoder(input_shape)
    decoder = build_decoder()
    autoencoder = build_autoencoder(encoder, decoder, input_shape)

    compile_autoencoder(autoencoder)
    history = autoencoder.fit(X_train, y_train, epochs=30, batch_size=15)
    return history


def main():
    X_train = np.random.rand(25, 544, 544, 3)
    y_train = np.random.rand(25, 544, 544, 3)
    input_shape = (544, 544, 3)

if __name__ == '__main__':
    main()
