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
from keras.applications.resnet import ResNet50
from keras.applications.vgg16 import VGG16
from make_model_V1 import make_resNet_model, make_with_resBlocks
from make_model_V2 import Emiqa


def train(samples, base_model_name, batch_size, num_class, train_dir, epochs_train,
          available_weights, crop_size, learning_rate):
    model = make_resNet_model(num_class=num_class)
    # model = make_with_resBlocks(num_class=num_class, num_filter=16, num_block=3)


    if available_weights is not None:
        model.load_weights(available_weights)

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

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=mse)
    model.summary()
    model.fit_generator(
        generator=train_generator,
        validation_data=validation_generator,
        epochs=epochs_train,
        verbose=1,
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
    parser.add_argument('--epochs_train', default=10)
    parser.add_argument('--learning_rate', default=0.001)
    parser.add_argument('--available_weights', default=None)
    parser.add_argument('--crop_size', default=[224,224])
    args = parser.parse_args()

    log_dir = os.path.join(args.train_dir, 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    weight_dir = os.path.join(args.train_dir, 'weights')
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
    with open(args.data_dir, 'r') as file:
        all_samples = json.load(file)
    
    train(samples=all_samples, base_model_name=args.base_model_name, batch_size=args.batch_size, num_class=args.num_class,
          train_dir=args.train_dir, epochs_train=args.epochs_train, available_weights=args.available_weights, crop_size=args.crop_size,
          learning_rate=args.learning_rate)
