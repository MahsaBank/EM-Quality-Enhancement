import argparse
import json
import os

import keras
from tensorflow.python.keras import backend as k
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from utilities import DataSequence
from matplotlib import cm
from keras.optimizers import Adam
from keras.losses import mse
from keras.layers import Flatten, Dense, LeakyReLU
from keras.applications.resnet import ResNet50
from keras.applications.vgg16 import VGG16


colmap = cm.get_cmap('viridis', 256)
np.savetxt('cmap.csv', (colmap.colors[...,0:3]*255).astype(np.uint8), fmt='%d', delimiter=',')


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


def make_resNet_model(activation_function, num_class, num_block, num_filter):
    # input_x = tf.keras.layers.Input((224, 224, 3))
    # x = identity_block(input_x, filter=num_filter)
    # for i in range(num_block-1):
    #     x = identity_block(x, filter=num_filter)
    # x = Flatten()(x)
    # x = Dense(units=512, activation='linear')(x)
    # x = LeakyReLU()(x)
    # x = Dense(units=10, activation='linear')(x)
    # x = LeakyReLU()(x)
    # x = Dense(units=num_class, activation='linear')(x)
    # model = tf.keras.models.Model(inputs=input_x, outputs=x, name='ResNetBlock')
    model = keras.Sequential()
    model.add(ResNet50(include_top=False, classes=num_class, input_shape=(224, 224, 3), weights='imagenet'))
    model.add(Flatten())
    model.add(Dense(units=512, activation='linear'))
    model.add(LeakyReLU())
    model.add(Dense(units=10, activation='linear'))
    model.add(LeakyReLU())
    model.add(Dense(units=num_class, activation='linear'))

    return model


def train(samples, base_model_name, batch_size, num_class, train_dir, epochs_train_all, epochs_train_dense,
          available_weights, crop_size, do_multi_process, learning_rate_all, activation_function):
    resnetmodel = make_resNet_model(activation_function=activation_function, num_class=num_class, num_filter=16, num_block=3)


    if available_weights is not None:
        resnetmodel.load_weights(available_weights)

    train_samples, test_samples = train_test_split(samples, test_size=0.05, shuffle=True, random_state=42)
    train_images_name = [train_samples[i]['image_name'] for i in range(len(train_samples))]
    train_labels = [train_samples[i]['label'] for i in range(len(train_samples))]
    train_generator = DataSequence(train_images_name, train_labels, batch_size, num_class)
    test_images_name = [test_samples[i]['image_name'] for i in range(len(test_samples))]
    test_labels = [test_samples[i]['label'] for i in range(len(test_samples))]
    validation_generator = DataSequence(test_images_name, test_labels, batch_size, num_class)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(train_dir, 'logs'))

    checkpoint_name = ('weights_'+base_model_name+'_{epoch:02d}_{loss:.3f}.hdf5')
    checkpoint_filepath = os.path.join(train_dir, 'weights', checkpoint_name)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        verbose=1,
        save_best_only=True)

    resnetmodel.compile(optimizer=Adam(learning_rate=learning_rate_all), loss=mse)
    resnetmodel.summary()
    resnetmodel.fit_generator(
        generator=train_generator,
        validation_data=validation_generator,
        epochs=epochs_train_all,
        verbose=1,
        use_multiprocessing=do_multi_process,
        workers=2,
        max_queue_size=30,
        callbacks=[tensorboard_callback, model_checkpoint_callback],)

    k.clear_session()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='train the Emiqa model for assessing the quality of EM images')
    parser.add_argument("--base_model_name", default='VGG16', type=str, help='It can be InceptionV3, and ...')
    parser.add_argument('--num_class', type=int, default=4, help='the number of quality classes including good, very good, bad,'
                                                     ' and very bad')
    parser.add_argument('--data_dir', default='./EM_quality_data.json', help='path/to/the/data/json/file')
    parser.add_argument('--train_dir', default='./', help='path/to/the/weights/and/logs/directory')
    parser.add_argument('--batch_size', default=20)
    parser.add_argument('--epochs_train_dense', default=10)
    parser.add_argument('--epochs_train_all', default=10)
    parser.add_argument('--learning_rate_dense', default=0.0001)
    parser.add_argument('--learning_rate_all', default=0.001)
    parser.add_argument('--available_weights', default=None)
    parser.add_argument('--crop_size', default=[224,224])
    parser.add_argument('--do_multiprocessing',default=False)
    parser.add_argument('--activation', default='softmax')
    args = parser.parse_args()

    log_dir = os.path.join(args.train_dir, 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    weight_dir = os.path.join(args.train_dir, 'weights')
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
    with open(args.data_dir, 'r') as file:
        all_samples = json.load(file)
    # train(**args.__dict__)
    train(samples=all_samples, base_model_name=args.base_model_name, batch_size=args.batch_size, num_class=args.num_class,
          train_dir=args.train_dir, epochs_train_all=args.epochs_train_all, epochs_train_dense=args.epochs_train_dense,
          available_weights=args.available_weights, crop_size=args.crop_size, do_multi_process=args.do_multiprocessing,
          learning_rate_all=args.learning_rate_all, activation_function=args.activation)