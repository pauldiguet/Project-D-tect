from tensorflow import keras
from keras import Sequential, layers, models
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D, Input
from keras.models import Model
import numpy as np
from dtect.Data_preparation.preprocessing import cropped_resized_images



def build_encoder(input_shape= (544,544,3)):

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
    decoder.add(Conv2DTranspose(512, (3, 3), padding='same', activation='relu', input_shape=build_encoder().layers[-1].output_shape[1:]))
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


def build_autoencoder(encoder, decoder, input_shape= (544,544,3)):
    inp = Input(input_shape)
    encoded = encoder(inp)
    decoded = decoder(encoded)
    autoencoder = Model(inp, decoded)
    return autoencoder


def compile_autoencoder(autoencoder):
    autoencoder.compile(loss='mean_squared_error',
                  optimizer='adam',
                    metrics=['accuracy'])


def train_autoencoder(df, input_shape= (544,544,3)):

    encoder = build_encoder(input_shape)
    decoder = build_decoder()
    autoencoder = build_autoencoder(encoder, decoder, input_shape)

    compile_autoencoder(autoencoder)
    print(np.array(df['image_x']).type)
    history = autoencoder.fit(np.array(df['image_x']), df['image_y'], epochs=1, batch_size=0)
    return history


if __name__ == '__main__':
    train_autoencoder(df=cropped_resized_images(train=True, category=1), input_shape= (544,544,3))
