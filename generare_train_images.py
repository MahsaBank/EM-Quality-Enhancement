import PIL
from PIL import Image
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras as k
import os

PIL.Image.MAX_IMAGE_PIXELS = 500000000


def extract_patches_and_augment(img, stride, w_path=None, tile_name=None, patch_size=[224, 224], do_save=True):
    height, width, _ = np.shape(img)
    patches = []
    datagen_flip = ImageDataGenerator(horizontal_flip=True)
    datagen_brightness = ImageDataGenerator(brightness_range=[0.4, 0.6])
    datagen_brightness_flip = ImageDataGenerator(horizontal_flip=True, brightness_range=[0.4, 0.6])
    data = []

    for y in range(0, height - patch_size[0] + 1, stride[0]):
        for x in range(0, width - patch_size[1] + 1, stride[1]):
            patch = img[y:y + patch_size[0], x:x + patch_size[1], :]
            patch = 255 - patch
            # data.append(patch)
            # data = np.asarray(data)
            # flipped_patch = datagen_flip.flow(data)
            # bright_patch = datagen_brightness.flow(data)
            # bright_flipped_patch = datagen_brightness_flip.flow(data)
            if do_save:
                k.utils.save_img(
                    os.path.join(w_path, os.path.splitext(tile_name)[0] + '_y' + str(y) + '_x' + str(x) + '.png'),
                    patch)
                # k.utils.save_img(
                #     os.path.join(w_path,
                #                  os.path.splitext(tile_name)[0] + '_inverted' + '_y' + str(y) + '_x' + str(x) + '.png'),
                #     inverted_patch)
                # k.utils.save_img(
                #     os.path.join(w_path,
                #                  os.path.splitext(tile_name)[0] + '_flipped' + '_y' + str(y) + '_x' + str(x) + '.png'),
                #     flipped_patch)
                # k.utils.save_img(
                #     os.path.join(w_path,
                #                  os.path.splitext(tile_name)[0] + '_brightness' + '_y' + str(y) + '_x' + str(x) +
                #                  '.png'),
                #     bright_patch)
                # k.utils.save_img(
                #     os.path.join(w_path,
                #                  os.path.splitext(tile_name)[0] + '_flipped_brightness' + '_y' + str(y) + '_x' + str(x)
                #                  + '.png'),
                #     bright_flipped_patch)
            # patches.append(patch)
            # patches.append(inverted_patch)
            # patches.append(flipped_patch)
            # patches.append(bright_patch)
            # patches.append(bright_flipped_patch)

    return patches


read_path = "\\\\storage1.ris.wustl.edu\wucci\Active\Merlin\Morgan Lab\MasterRaw\KxR_1A\Copy of Raws\w8\w8_2_Sec044_Montage\Tile_r3-c2_w8_2_sec044.tif"
img = Image.open(read_path)
img = img.convert('RGB')
img = np.asarray(img)
img = img[6000:12000, 6000:12000, :]
extract_patches_and_augment(img, stride=[150, 150], w_path="D:\Quality_Groups\\revised-groups\Q7_512",
                            tile_name='Tile_r3-c2_w8_2_sec044.tif', patch_size=[512, 512])
