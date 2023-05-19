
import tensorflow as tf
import math
import numpy as np
from matplotlib import cm


def create_colmap():
    colmap = cm.get_cmap('viridis', 256)
    np.savetxt('cmap.csv', (colmap.colors[...,0:3]*255).astype(np.uint8), fmt='%d', delimiter=',')

    
def crop_image(img, crop_size=[224, 224]):
    img_shape = np.shape(img)
    i, j = np.random.randint(0, img_shape[0]-crop_size[0]+1), np.random.randint(0, img_shape[1]-crop_size[1]+1)
    img_arr = np.array(img)
    return img_arr[i:i+crop_size[0], j:j+crop_size[1], :]


def get_data(x_set, y_set, crop_size, shufffle=False):
    dat_len = len(x_set)
    indexes = [i for i in range(dat_len)]
    if shuffle:
        np.random.shuffle(indexes)
    batch_x = [x_set[i] for i in indexes]
    image_x = np.empty((len(batch_x), *crop_size, 3))
    if y_set is not None:
        label_y = np.array([y_set[i] for i in indexes])
    else:
        label_y = []
    i = 0
    for file_name in batch_x:
        img = tf.keras.utils.load_img(file_name)
        img = crop_image(img, crop_size)
        if img is not None:
            image_x[i, ] = img
            i += 1
    return image_x, label_y


class DataSequence(tf.keras.utils.Sequence):
    def __init__(self, x_set, y_set, batch_size=20, num_class=1, crop_size=[224, 224], shuffle=True):
        self.x_set = x_set
        self.y_set = y_set
        self.batch_size = batch_size
        self.crop_size = crop_size
        self.shuffle = shuffle
        self.datalen = len(self.x_set)
        self.indexes = np.arange(self.datalen)
        self.num_class = num_class
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(self.datalen/self.batch_size)

    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = [self.x_set[i] for i in batch_indexes]
        image_x = np.empty((len(batch_x), *self.crop_size, 3))
        batch_y = [self.y_set[i] for i in batch_indexes]
        label_y = np.zeros((len(batch_y), self.num_class))
        i = 0

        for file_name in batch_x:
            img = tf.keras.utils.load_img(file_name)
            img = crop_image(img, self.crop_size)
            if img is not None:
                image_x[i, ] = img
                if self.num_class > 1:
                    label_y[i, batch_y[i]] = 1
                else:
                    label_y[i] = batch_y[i]
                i += 1
        return image_x, label_y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)






