import argparse

import h5py
import os
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Train Data info')
parser.add_argument('--nb_train_all', type=int)
parser.add_argument('--nb_test_all', type=int)
parser.add_argument('--source', type=str)
parser.add_argument('--output', type=str)

args = parser.parse_args()

nb_train_all = args.nb_train_all
nb_test_all = args.nb_test_all
img_count = nb_train_all + nb_test_all

source = args.source
output = args.output
data = h5py.File(os.path.join(output, 'shoes.h5'), 'w-')
dt = h5py.special_dtype(vlen=np.dtype('uint8'))
data.create_dataset(name='x', shape=(img_count,), dtype=dt)
data.create_dataset(name='y', shape=(img_count, 1), dtype='int32')

ix = 0

##train
with open(
        os.path.join(
            source,
            'Shoes_train.txt'
        )
) as f:
    f.readline()
    nb_images = 0

    for (image_id, class_id, _, path) in tqdm(map(str.split, f), total=nb_train_all):
        nb_images += 1
        img_path = os.path.join(source, path)

        c_f = open(img_path, 'rb')
        img_bytes = c_f.read()
        c_f.close()

        data['x'][ix] = np.fromstring(img_bytes, dtype='uint8')
        data['y'][ix] = int(class_id) - 1

        ix += 1

    assert nb_images == nb_train_all

##test
with open(
        os.path.join(
            source,
            'Shoes_test.txt'
        )
) as f:
    f.readline()
    nb_images = 0

    for (image_id, class_id, _, path) in tqdm(map(str.split, f), total=nb_test_all):
        nb_images += 1
        img_path = os.path.join(source, path)

        c_f = open(img_path, 'rb')
        img_bytes = c_f.read()
        c_f.close()

        data['x'][ix] = np.fromstring(img_bytes, dtype='uint8')
        data['y'][ix] = int(class_id) - 1

        ix += 1

    assert nb_images == nb_test_all

data.close()
