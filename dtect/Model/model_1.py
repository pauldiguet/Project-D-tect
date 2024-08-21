import tensorflow as tf

from tensorflow import keras
from keras import Sequential, layers, models
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Conv2DTranspose, Input
from keras.models import Model


input_shape = (544,544,3)

def build_encoder(input_shape):

    encoder = Sequential()

    encoder.add(layers.Conv2D(32, (3,3) , strides=(2,2), input_shape=input_shape, padding='same', activation='relu'))
    encoder.add(layers.MaxPooling2D(2,2))

    encoder.add(layers.Conv2D(64, (3,3),  strides=(2,2), padding='same', activation='relu'))
    encoder.add(layers.MaxPooling2D(2,2))

    encoder.add(layers.Conv2D(128, (3,3),  strides=(2,2), padding='same', activation='relu'))
    encoder.add(layers.MaxPooling2D(2,2))

    encoder.add(layers.Conv2D(256, (3,3),  strides=(2,2), padding='same', activation='relu'))

    return encoder



def build_decoder():

    decoder = Sequential()

    decoder.add(Conv2DTranspose(128, (3,3), strides=(2,2), padding='same', activation='relu'))

    decoder.add(Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', activation='relu'))

    decoder.add(Conv2DTranspose(32, (3,3), strides=(2,2), padding='same', activation='relu'))

    decoder.add(Conv2DTranspose(3, (3,3), strides=(2,2), padding='same', activation='sigmoid'))

    return decoder


def build_autoencoder(encoder, decoder):
    inp = Input(input_shape)
    encoded = encoder(inp)
    decoded = decoder(encoded)
    autoencoder = Model(inp, decoded)
    return autoencoder


def compile_autoencoder(autoencoder):
    autoencoder.compile(loss='binary_crossentropy',
                  optimizer='adam',
                    metrics=['accuracy'])
    return autoencoder


def main():
    encoder = build_encoder(input_shape)
    decoder = build_decoder()
    autoencoder = build_autoencoder(encoder, decoder)
    compile_autoencoder(autoencoder)


if __name__ == '__main__':
    main()
