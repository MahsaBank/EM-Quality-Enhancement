import argparse
import json
import os.path
import keras
import numpy as np
import PIL.Image
from tensorflow import keras as k
import time
import multiprocessing
from utilities import extract_patches

patch_size = [512, 512]
do_save = False

PIL.Image.MAX_IMAGE_PIXELS = 500000000


def calculate_qs(tile, model):
    quality_scores = []
    tile_name = tile['image_name']
    img = k.utils.load_img(tile_name)
    height, width, _ = np.shape(img)
    img = 255 - np.asarray(img)
    patches, indexes = extract_patches(img=img, w_path=None, tile_name=tile_name, patch_size=patch_size,
                                       stride=[round(height // 2), round(width // 2)], do_save=do_save)

    data = np.asarray(patches)
    predictions = model.predict(data)
    for itr in range(len(predictions)):
        prediction = str(predictions[itr])
        y_ind = str(indexes[itr]['y'])
        x_ind = str(indexes[itr]['x'])
        quality_scores.append({'image_name': tile_name, 'y': y_ind, 'x': x_ind, 'quality_score': prediction})

    return quality_scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This script calculates the quality scores for all tiles listed in the'
                                                 'input json file and create a json file including the quality scores')
    parser.add_argument("--tiles_json_file", default='C:\\Users\\Mahsa\\PycharmProjects\\Emiqa\\EM_tiles_list_1.json', help='path/to/tiles/json/file')
    parser.add_argument("--write_path", default="C:\\Users\\Mahsa\\PycharmProjects\\Emiqa",
                        help='where/to/write/the/output/json/file')
    parser.add_argument('--output_name', default='quality_scores_1')
    parser.add_argument('--model_file',
                        default="/storage1.ris.wustl.edu/jlmorgan/Active/mahsa/containers/Emiqa/weights/model_renet_size_512_02_0.002.h5")

    args = parser.parse_args()

    w_path = args.write_path

    t1 = time.time()
    with open(args.tiles_json_file, 'r') as file:
        tiles_list = json.load(file)

    model = keras.models.load_model(args.model_file)
    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_processes)

    all_quality_scores = pool.starmap(calculate_qs, [(tile, model) for tile in tiles_list])
    pool.close()
    pool.join()

    out_path = os.path.join(w_path, args.output_name + '.json')
    out_file = open(out_path, 'w')
    t2 = time.time()
    print('total time is :', str(t2 - t1), ' secs')

    json.dump(all_quality_scores, out_file)
    out_file.close()
