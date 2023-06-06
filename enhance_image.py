import os.path
import keras
import argparse
import json
import PIL.Image
from PIL import Image
import numpy as np
from tools.utilities import preprocess_image, deprocess_image, extract_patches, stich_patches, rgb_to_gray, \
    process_patches, gray_to_rgb
from deblurgan.model import generator_model

PIL.Image.MAX_IMAGE_PIXELS = 500000000


def improve_using_gan(weights_file, quality_file, write_path, subtile_position, quality_threshold, patch_size):
    with open(quality_file, 'r') as file:
        all_tiles = json.load(file)
    tiles_for_improving = [tile["image_name"] for tile in all_tiles if float(tile["quality_score"]) < quality_threshold]
    model = generator_model()
    model.load_weights(weights_file)
    for tile in tiles_for_improving:
        basename = os.path.basename(tile)
        base_filename = os.path.splitext(basename)[0] + '_x' + str(subtile_position[0]) + '-' + str(
            subtile_position[1]) + '_y' \
                        + str(subtile_position[2]) + '-' + str(subtile_position[3]) + os.path.splitext(basename)[1]
        img = Image.open(tile)
        img = gray_to_rgb(img)
        # img = img.convert('RGB')
        img = np.asarray(img)

        if subtile_position is not None:
            subimg = img[subtile_position[0]:subtile_position[1], subtile_position[2]:subtile_position[3], :]
        else:
            subimg = img
        filename = 'original_' + base_filename
        original_subimg = 255 - subimg
        original_subimg = Image.fromarray(original_subimg)
        original_subimg = original_subimg.convert('L')
        original_subimg.save(os.path.join(write_path, filename))
        # subimg_shape = subimg.shape
        # remainder = subimg_shape[0] % patch_size
        # if remainder != 0:
        #     padding_width = patch_size - remainder
        #     subimg = np.pad(subimg, padding_width, mode='constant')
        #     # subimg = pad_rgb_image(subimg, padding_width)
        # filename = 'after_padding_' + base_filename
        # subimg = Image.fromarray(subimg)
        # subimg.save(os.path.join(write_path, filename))
        # subimg = subimg.convert('RGB')
        # subimg = gray_to_rgb(subimg)
        # subimg = np.asarray(subimg)
        # subimg.setflags(write=1)
        # subimg = preprocess_image(subimg)
        patches, indexes = extract_patches(subimg, patch_size=[patch_size, patch_size], stride=[patch_size, patch_size])
        patches = process_patches(patches, preprocess=1)
        improved_patches = model.predict(x=np.asarray(patches))
        improved_patches = process_patches(improved_patches, preprocess=0)
        improved_subimg = stich_patches(improved_patches, indexes, subimg, patch_size)

        filename = 'improved_' + base_filename

        # improved_subimg = deprocess_image(improved_subimg)

        improved_subimg = 255 - improved_subimg
        improved_subimg = Image.fromarray(improved_subimg)
        improved_subimg = improved_subimg.convert('L')
        improved_subimg.save(os.path.join(write_path, filename))


if __name__ == '__main__':
    pars = argparse.ArgumentParser()
    pars.add_argument('--weights_file',
                      default="Z:\Active\mahsa\containers\DeblurGAN\deblur-gan\weights\\210\generator_7_244.h5")
    pars.add_argument('--quality_scores', default='C:/Users/Mahsa/PycharmProjects/Emiqa/quality_scores.json')
    pars.add_argument('--output_path', default='C:/Users/Mahsa/PycharmProjects/Emiqa/test_images')
    pars.add_argument('--subtile_position', default=None)
    pars.add_argument('--quality_threshold', default=0.5, type=float)
    pars.add_argument('--patch_size', default=256, type=int)
    args = pars.parse_args()

    improve_using_gan(weights_file=args.weights_file, quality_file=args.quality_scores, write_path=args.output_path,
                      subtile_position=args.subtile_position, quality_threshold=args.quality_threshold,
                      patch_size=args.patch_size)
