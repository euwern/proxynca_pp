import h5py
import os 
import numpy as np
from tqdm import tqdm
import torchvision
import scipy.io

root = '/mnt/datasets/cars196_alt'
annos_fn = 'cars_annos.mat'
cars = scipy.io.loadmat(os.path.join(root, annos_fn))
ys = [int(a[5][0] - 1) for a in cars['annotations'][0]]
im_paths = [a[0][0] for a in cars['annotations'][0]]

img_count = len(ys)

data = h5py.File(os.path.join(root, 'cars.h5'), 'w')
dt = h5py.special_dtype(vlen=np.dtype('uint8'))
data.create_dataset(name='x', shape=(img_count, ), dtype=dt)
data.create_dataset(name='y', shape=(img_count, 1), dtype='int32')

print(img_count)

ix = 0
for im_path, y in tqdm(zip(im_paths, ys)):
    #if y in classes: # choose only specified classes
    img_path = os.path.join(root, im_path)

    c_f = open(img_path, 'rb')   
    img_bytes = c_f.read()
    c_f.close()

    data['x'][ix] = np.fromstring(img_bytes, dtype='uint8')
    data['y'][ix] = y
    
    ix += 1
   

data.close()
