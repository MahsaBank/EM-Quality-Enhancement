
import keras
from keras.layers import Flatten, Dense, LeakyReLU
from keras.applications.resnet import ResNet50
import tensorflow as tf



def identity_block(x, filter):
    x_skip = x
    # layer 1
    x = tf.keras.layers.Conv2D(filter, (3, 3), padding='same', strides=(2, 2))(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)
    # layer 2
    x = tf.keras.layers.Conv2D(filter, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    # add residue
    x_skip = tf.keras.layers.Conv2D(filter, (1, 1), strides=(2, 2))(x_skip)
    x = tf.keras.layers.Add()([x, x_skip])
    x = tf.keras.layers.Activation('relu')(x)
    return x


def make_resNet_model(num_class):
    model = keras.Sequential()
    model.add(ResNet50(include_top=False, classes=num_class, input_shape=(224, 224, 3), weights='imagenet'))
    model.add(Flatten())
    model.add(Dense(units=512, activation='linear'))
    model.add(LeakyReLU())
    model.add(Dense(units=10, activation='linear'))
    model.add(LeakyReLU())
    model.add(Dense(units=num_class, activation='linear'))

    return model


def make_with_resBlocks(num_class, num_block, num_filter):
    input_x = tf.keras.layers.Input((224, 224, 3))
    x = identity_block(input_x, filter=num_filter)
    for i in range(num_block-1):
        x = identity_block(x, filter=num_filter)
    x = Flatten()(x)
    x = Dense(units=512, activation='linear')(x)
    x = LeakyReLU()(x)
    x = Dense(units=10, activation='linear')(x)
    x = LeakyReLU()(x)
    x = Dense(units=num_class, activation='linear')(x)
    model = tf.keras.models.Model(inputs=input_x, outputs=x, name='ResNetBlock')

    return model