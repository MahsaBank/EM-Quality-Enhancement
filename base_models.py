
from keras.models import Model
from keras.applications.resnet import ResNet50
from keras.layers import GlobalAvgPool2D, Flatten, Dense, LeakyReLU


def customized_renet_model(input_shape=(512, 512, 3), weights='imagenet'):
    base_model = ResNet50(input_shape=input_shape, include_top=False, weights=weights)
    x = base_model.output
    x = GlobalAvgPool2D()(x)
    x = Flatten()(x)
    x = Dense(units=512, activation='linear')(x)
    x = LeakyReLU()(x)
    x = Dense(units=10, activation='linear')(x)
    x = LeakyReLU()(x)
    x = Dense(units=1, activation='linear')(x)
    x = LeakyReLU()(x)
    model = Model(inputs=base_model.input, outputs=x)
    return model


model = customized_renet_model()
model.load_weights("\\\\storage1.ris.wustl.edu\jlmorgan\Active\mahsa\containers\Emiqa\weights\weights_resnet_with_512size_03_0.001.hdf5")
model.save("/storage1.ris.wustl.edu/jlmorgan/Active/mahsa/containers/Emiqa/weights/model_resnet_with_512size_03_0.001.hd")
