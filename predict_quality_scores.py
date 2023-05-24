import argparse
import json
import os.path
import keras
import numpy as np
import PIL.Image
from tensorflow import keras as k


PIL.Image.MAX_IMAGE_PIXELS = 500000000


def extract_patches(img, stride, w_path, tile_name, patch_size=[224, 224], do_save=False):
    height, width, _ = np.shape(img)
    patches = []

    for y in range(0, height - patch_size[0] + 1, stride[0]):
        for x in range(0, width - patch_size[1] + 1, stride[1]):
            patch = img[y:y + patch_size[0], x:x + patch_size[1]]
            if do_save:
                k.utils.save_img(os.path.join(w_path, os.path.splitext(tile_name)[0]+'_y'+str(y)+'_x'+str(x)+'.png'),
                                 patch)
            patches.append(patch)

    return patches


def calculate_qs(tiles_json_file, w_path, json_name, model_file):
    with open(tiles_json_file, 'r') as file:
        tiles_list = json.load(file)
    quality_scores = []
    status = -1
    for tile in tiles_list:
        img = k.utils.load_img(tile['image_name'])
        height, width, depth = np.shape(img)
        img = 256 - np.asarray(img)
        patches = extract_patches(img=img, w_path=w_path, tile_name=tile['image_name'],
                                  stride=[round(height // 3), round(width // 3)], do_save=False)
        model = keras.models.load_model(model_file)
        data = np.asarray(patches)
        predictions = model.predict(data)
        quality_scores.append({'image_name': tile['image_name'], 'quality_score': str(np.mean(predictions))})

    out_path = os.path.join(w_path, json_name+'.json')
    out_file = open(out_path, 'w')
    json.dump(quality_scores, out_file)
    out_file.close()
    status = 2

    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This script calculates the quality scores for all tiles listed in the'
                                                 'input json file and create a json file including the quality scores')
    parser.add_argument("--tiles_json_file", default='./EM_tiles_list.json', help='path/to/tiles/json/file')
    parser.add_argument("--write_path", default='./', help='where/to/write/the/output/json/file')
    parser.add_argument('--output_name', default='quality_scores')
    parser.add_argument('--model_file', default="Z:/Active/mahsa/containers/Emiqa/resnet50ModelforMatlab.h5")

    args = parser.parse_args()

    calculate_qs(tiles_json_file=args.tiles_json_file, w_path=args.write_path, json_name=args.output_name, model_file=args.model_file)