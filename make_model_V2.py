

import importlib
from keras.models import Model
from keras.layers import Dropout, Dense
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy, mse
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.applications.mobilenet_v3 import MobileNetV3Small
from keras.applications.resnet import ResNet50


class Emiqa:
    def __init__(self, base_model_name='vgg16', num_class=4, loss=categorical_crossentropy, weights='imagenet',
                 learning_rate=0.001, drop_rate=0, activation_function='softmax'):
        self.num_class = num_class
        self.base_model_name = base_model_name
        self.loss = loss
        self.weights = weights
        self.learning_rate = learning_rate
        self.drop_rate = drop_rate
        self.activation = activation_function

    def create(self):

        self.base_model = VGG16(input_shape=(224, 224, 3), include_top=False, weights=self.weights, pooling='avg')
        print(self.base_model.output)
        x = Dropout(self.drop_rate)(self.base_model.output)
        x = Dense(units=self.num_class, activation=self.activation)(x)

        self.Emiqa_model = Model(self.base_model.inputs, x)

    def compile(self):
        self.Emiqa_model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss=self.loss)
