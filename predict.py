
import os
import glob
import json
import argparse
import time
import numpy as np
from tools/make_model_v2 import Emiqa
from tools/utilities import DataSequence
from tools/model_make_v1 import make_resNet_model
import tensorflow as tf
import autokeras as ak
from tools/utilities import get_data


def main(base_model_name, weights_file, image_dir, predictions_file, model_file, img_format):
    
    test_images_name = glob.glob(os.path.join(image_dir, '*.'+img_format))

    # build model and load weights
    
    #model = tf.keras.models.load_model(model_file,custom_objects=ak.CUSTOM_OBJECTS)
    # emiqa = Emiqa(base_model_name=base_model_name, weights=None, num_class=1)
    # emiqa.create()
    # emiqa.Emiqa_model.load_weights(weights_file)
    # model = emiqa.Emiqa_model
    
    model = make_resNet_model(num_class=1)
    model.load_weights(weights_file)
    # print(model.summary())
    
    test_data = get_data(test_images_name, y_set=None, crop_size=[224,224], shuffle=False)
  
    # get predictions
    
    time1 = time.time()
    predictions = model.predict(test_data)
    time2 = time.time()
    total_time = time2 - time1
    # print("total time is:",total_time)
    
    np.savetxt(predictions_file+'predictions_'+base_model_name+'.txt', predictions, fmt='%.3f')
    np.savetxt(predictions_file+'samples_'+base_model_name+'.txt', samples, '%s')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--base-model-name', help='CNN base model name', default='resnet')
    parser.add_argument('-w', '--weights-file', help='path of weights file', default='/storage1/fs1/jlmorgan/Active/mahsa/containers/Emiqa/weights/weights_resnet_09_0.023.hdf5')
    parser.add_argument('-is', '--image-dir', help='image directory', default='/storage1/fs1/jlmorgan/Active/mahsa/quality_groups/test2')
    parser.add_argument('-pf', '--predictions-file', help='file with predictions', required=False, default='/storage1/fs1/jlmorgan/Active/mahsa/containers/Emiqa/predictions/')
    parser.add_argument('-m', '--model-file', help='path of model file', default="/storage1/fs1/jlmorgan/Active/mahsa/containers/Emiqa/auto_model_NormalizationConvBlockResnetXceptionDense/best_model")
    parser.add_argument('-if', '--image-format', default='.tif')

    args = parser.parse_args()

    main(**args.__dict__)
