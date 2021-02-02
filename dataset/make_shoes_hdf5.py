import h5py
import os 
import numpy as np
from tqdm import tqdm

nb_train_all = 66812
nb_test_all = 66653
img_count = nb_train_all + nb_test_all


source = '/home/ubuntu/workstation/datasets/shoe'
output = '/home/ubuntu/workstation/proxynca'
data = h5py.File(os.path.join(output, 'shoes.h5'), 'w-')
dt = h5py.special_dtype(vlen=np.dtype('uint8'))
data.create_dataset(name='x', shape=(img_count, ), dtype=dt)
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
        #print(int(class_id) - 1)
        data['y'][ix] = int(class_id) - 1
        #print(ix, data['y'][ix])
        
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

        #print(int(class_id) - 1)
        data['x'][ix] = np.fromstring(img_bytes, dtype='uint8')
        data['y'][ix] = int(class_id) - 1
        
        ix += 1
    
    assert nb_images == nb_test_all

data.close()

