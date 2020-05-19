import h5py
import os 
import numpy as np
from tqdm import tqdm
import torchvision
import scipy.io

root = '/mnt/datasets/inshop'

####
with open(
    os.path.join(
        root, 'Eval/list_eval_partition.txt'
    ), 'r'
) as f:
    lines = f.readlines()

# store for using later '__getitem__'

nb_samples = int(lines[0].strip('\n'))
#print(nb_samples)
assert nb_samples == 52712

or_im_paths = {'train': [], 'query': [], 'gallery': []}
or_ys = {'train': [], 'query': [], 'gallery': []}
or_I_ = {'train': 0, 'query': 0, 'gallery': 0}
or_I = {'train': [], 'query': [], 'gallery': []}

# start from second line, since 0th and 1st contain meta-data
for line in lines[2:]:
    im_path, im_id, eval_type = [
        l for l in line.split(' ') if l != '' and l != '\n']
    y = int(im_id.split('_')[1])


    # this is the old code chunk
    or_im_paths[eval_type] += [os.path.join(root, im_path)]
    or_ys[eval_type] += [y]
    or_I[eval_type] += [or_I_[eval_type]]
    or_I_[eval_type] += 1

nb_samples_counted = len(or_im_paths['train']) + \
        len(or_im_paths['gallery']) + len(or_im_paths['query'])
assert nb_samples_counted == nb_samples

# verify that labels are sorted for next step
or_ys['query'] == sorted(or_ys['query'])
or_ys['gallery'] == sorted(or_ys['gallery'])

assert len(or_ys['train']) == 25882
assert len(or_ys['query']) == 14218
assert len(or_ys['gallery']) == 12612

# verify that query and gallery have same labels
assert set(or_ys['query']) == set(or_ys['gallery'])

# labels of query and gallery are like [1, 1, 7, 7, 8, 11, ...]
# condense them such that ordered without spaces,
# i.e. 1 -> 1, 7 -> 2, ...
idx_to_class = {idx: i for i, idx in enumerate(
    sorted(set(or_ys['query']))
)}
for _type in ['query', 'gallery']:
    or_ys[_type] = list(
        map(lambda x: idx_to_class[x], or_ys[_type]))

# same thing for train labels
idx_to_class = {idx: i for i, idx in enumerate(
    sorted(set(or_ys['train']))
)}
or_ys['train'] = list(
    map(lambda x: idx_to_class[x], or_ys['train']))

# should be 3997 classes for training, 3985 for query/gallery
assert len(set(or_ys['train'])) == 3997
assert len(set(or_ys['query'])) == 3985
assert len(set(or_ys['gallery'])) == 3985

###storing data to hdf5
train_count = 25882
query_count = 14218
gallery_count = 12612


data = h5py.File(os.path.join(root, 'inshop.h5'), 'w')
dt = h5py.special_dtype(vlen=np.dtype('uint8'))
data.create_dataset(name='train_x', shape=(train_count, ), dtype=dt)
data.create_dataset(name='train_y', shape=(train_count, 1), dtype='int32')
data.create_dataset(name='train_i', shape=(train_count, 1), dtype='int32')
data.create_dataset(name='query_x', shape=(query_count, ), dtype=dt)
data.create_dataset(name='query_y', shape=(query_count, 1), dtype='int32')
data.create_dataset(name='query_i', shape=(query_count, 1), dtype='int32')
data.create_dataset(name='gallery_x', shape=(gallery_count, ), dtype=dt)
data.create_dataset(name='gallery_y', shape=(gallery_count, 1), dtype='int32')
data.create_dataset(name='gallery_i', shape=(gallery_count, 1), dtype='int32')



def store_data(dset_type):
    print("======>>>", len(or_ys[dset_type]))
    for ix in tqdm(range(len(or_ys[dset_type]))):
        y = or_ys[dset_type][ix]
        im_path = or_im_paths[dset_type][ix]
        ii = or_I[dset_type][ix]
        
        c_f = open(im_path, 'rb')   
        img_bytes = c_f.read()
        c_f.close()

        data[dset_type + '_x'][ix] = np.fromstring(img_bytes, dtype='uint8')
        data[dset_type + '_y'][ix] = y
        data[dset_type + '_i'][ix] = ii

store_data(dset_type='train')
store_data(dset_type='query')
store_data(dset_type='gallery')

data.close()

