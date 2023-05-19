
import os
import glob
import json
import argparse
import time

#from utils.utils import calc_mean_score, save_json
#from handlers.model_builder import Nima
from handlers.data_generator import TestDataGenerator
from make_model import Emiqa
from utilities import DataSequence
#from train_unet import make_unet_regression_model
from train_ResNet import make_resNet_model
import tensorflow as tf
import autokeras as ak
from autoModel import get_data



def image_file_to_json(img_path):
    img_dir = os.path.dirname(img_path)
    img_id = os.path.basename(img_path).split('.')[0]

    return img_dir, [{'image_id': img_id}]


def image_dir_to_json(img_dir, img_type='tif'):
    img_paths = glob.glob(os.path.join(img_dir, '*.'+img_type))

    samples = []
    for img_path in img_paths:
        img_id = os.path.basename(img_path).split('.')[0]
        samples.append({'image_id': img_id})

    return samples


def predict(model, data_generator):
    return model.predict_generator(data_generator, workers=8, use_multiprocessing=True, verbose=1)


def main(base_model_name, weights_file, image_source, predictions_file, model_file, img_format='tif'):
    # load samples
    if os.path.isfile(image_source):
        image_dir, samples = image_file_to_json(image_source)
    else:
        image_dir = image_source
        samples = image_dir_to_json(image_dir, img_type='tif')

    # build model and load weights
    #model = tf.keras.models.load_model(model_file,custom_objects=ak.CUSTOM_OBJECTS)
    #model.load_weights(weights_file)
    #print(model.summary())
    model = make_resNet_model(activation_function='relu',num_class=1, num_block=3, num_filter=16)
    model.load_weights(weights_file)
    tf.keras.saving.save_model(model,'//storage1.ris.wustl.edu/jlmorgan/Active/mahsa/containers/Emiqa/resnet50ModelforMatlab.h5')

    print(model.summary())
    # emiqa = Emiqa(base_model_name=base_model_name, weights=None, num_class=1)
    # emiqa.create()
    # emiqa.Emiqa_model.load_weights(weights_file)


    test_images_name = [image_dir+'/'+samples[i]['image_id']+'.tif' for i in range(len(samples))]
    test_data = get_data(test_images_name, y_set=None, crop_size=[224,224])
  
    # initialize data generator
    data_generator = TestDataGenerator(samples, image_dir, 5, 1, None,
                                        img_format=img_format)
    # emiqa.preprocess()
    # get predictions
    time1 = time.time()
    predictions = predict(model, data_generator)
    # predictions = predict(emiqa.Emiqa_model, data_generator)
    time2 = time.time()
    total_time = time2 - time1
    print("total time is:",total_time)
    i=0
    for prediction in predictions:
        print('the prediction for test image ',samples[i],'is ',prediction)
        i=i+1

    # calc mean scores and add to samples
    for i, sample in enumerate(samples):
        sample['mean_score_prediction'] = calc_mean_score(predictions[i])

    print(json.dumps(samples, indent=2))

    if predictions_file is not None:
        save_json(samples, predictions_file)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--base-model-name', help='CNN base model name', default='InceptionV3')
    parser.add_argument('-w', '--weights-file', help='path of weights file', default="\\storage1.ris.wustl.edu\jlmorgan\Active\mahsa\containers\Emiqa\weights\weights_resnet_08_0.029.hdf5")
    parser.add_argument('-is', '--image-source', help='image directory or file', default='/storage1/fs1/jlmorgan/Active/mahsa/quality_groups/test2')
    parser.add_argument('-pf', '--predictions-file', help='file with predictions', required=False, default=None)
    parser.add_argument('-m', '--model-file', help='path of model file', default="//storage1.ris.wustl.edu/jlmorgan/Active/mahsa/containers/Emiqa/auto_model_NormalizationConBlockDence/best_model")

    args = parser.parse_args()

    main(**args.__dict__)
