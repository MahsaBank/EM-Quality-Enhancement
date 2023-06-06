import argparse
import json
import os.path
import keras
import numpy as np
import PIL.Image
from tensorflow import keras as k
import time
# from tools.utilities import extract_patches


PIL.Image.MAX_IMAGE_PIXELS = 500000000


def extract_patches(img, stride, w_path=None, tile_name=None, patch_size=[512, 512], do_save=True):
    if len(np.shape(img)) < 3:
        height, width = np.shape(img)
    else:
        height, width, _ = np.shape(img)
    patches = []
    indexes = []

    for y in range(0, height - patch_size[0] + 1, stride[0]):
        for x in range(0, width - patch_size[1] + 1, stride[1]):
            if len(np.shape(img)) < 3:
                patch = img[y:y + patch_size[0], x:x + patch_size[1]]
            else:
                patch = img[y:y + patch_size[0], x:x + patch_size[1], :]
            if do_save:
                k.utils.save_img(os.path.join(w_path, os.path.splitext(tile_name)[0]+'_y'+str(y)+'_x'+str(x)+'.png'),
                                 patch)
            patches.append(patch)
            indexes.append({'y': y, 'x': x})

    return patches, indexes


def calculate_qs(tiles_json_file, w_path, json_name, model_file):
    with open(tiles_json_file, 'r') as file:
        tiles_list = json.load(file)
    quality_scores = []
    status = -1
    for tile in tiles_list:
        img = k.utils.load_img(tile['image_name'])
        height, width, depth = np.shape(img)
        img = 255 - np.asarray(img)
        patches, indexes = extract_patches(img=img, w_path=w_path, tile_name=tile['image_name'],
                                  stride=[round(height // 2), round(width // 2)], do_save=True)
        model = keras.models.load_model(model_file)
        data = np.asarray(patches)
        predictions = model.predict(data)
        for itr in range(len(predictions)):
            prediction = str(predictions[itr])
            y_ind = str(indexes[itr]['y'])
            x_ind = str(indexes[itr]['x'])
            quality_scores.append({'image_name': tile['image_name'], 'y': y_ind, 'x': x_ind, 'quality_score': prediction})

    out_path = os.path.join(w_path, json_name+'.json')
    out_file = open(out_path, 'w')
    json.dump(quality_scores, out_file)
    out_file.close()
    status = 2

    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This script calculates the quality scores for all tiles listed in the'
                                                 'input json file and create a json file including the quality scores')
    parser.add_argument("--tiles_json_file", default='./EM_tiles_list_1.json', help='path/to/tiles/json/file')
    parser.add_argument("--write_path", default="C:\\Users\\Mahsa\\PycharmProjects\\Emiqa", help='where/to/write/the/output/json/file')
    parser.add_argument('--output_name', default='quality_scores_1')
    parser.add_argument('--model_file', default="/storage1.ris.wustl.edu/jlmorgan/Active/mahsa/containers/Emiqa/weights/model_renet_size_512_02_0.001.h5")

    args = parser.parse_args()

    t1 = time.time()
    calculate_qs(tiles_json_file=args.tiles_json_file, w_path=args.write_path, json_name=args.output_name, model_file=args.model_file)
    t2 = time.time()
    print('total time is: ', str(t2-t1))