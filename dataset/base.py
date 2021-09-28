from __future__ import print_function
from __future__ import division

import os
import torch
import PIL.Image
from distutils.dir_util import copy_tree
import io
import h5py
from shutil import copyfile
import time

'''
class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, root, classes, transform = None):
        self.classes = classes
        self.root = root
        self.transform = transform
        self.ys, self.im_paths, self.I = [], [], []

    def nb_classes(self):
        assert set(self.ys) == set(self.classes)
        return len(self.classes)

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, index):
        im = PIL.Image.open(self.im_paths[index])
        # convert gray to rgb
        if len(list(im.split())) == 1 : im = im.convert('RGB') 
        if self.transform is not None:
            im = self.transform(im)
        return im, self.ys[index], index

    def get_label(self, index):
        return self.ys[index]

    def set_subset(self, I):
        self.ys = [self.ys[i] for i in I]
        self.I = [self.I[i] for i in I]
        self.im_paths = [self.im_paths[i] for i in I]
'''


class BaseDataset_hdf5(torch.utils.data.Dataset):
    def __init__(self, root, source, classes, transform=None, prefix=''):
        self.classes = classes
        self.root = root
        self.transform = transform
        self.ys, self.im_paths, self.I = [], [], []
        self.data_h5 = None
        self.prefix = prefix

        # making sure it is not an old copy or broken copy
        if not os.path.exists(root):
            if not os.path.exists(os.path.dirname(root)):
                os.makedirs(os.path.dirname(root))

            print('copying file from source to root')
            print('from:', source)
            print('to:', root)
            c_time = time.time()
            copyfile(source, root)
            elapsed = time.time() - c_time
            print('done copying file: %.2fs' % elapsed)

    def nb_classes(self):
        assert set(self.ys) == set(self.classes)
        return len(self.classes)

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, index):
        # im = PIL.Image.open(self.im_paths[index])
        # if self.data_h5 is None:
        self.data_h5 = h5py.File(self.root, mode='r')

        curr_index = self.I[index]
        im = PIL.Image.open(io.BytesIO(self.data_h5[self.prefix + 'x'][curr_index]))
        self.data_h5.close()
        ''' 
        try:
            im = PIL.Image.open(io.BytesIO(self.data_h5['x'][curr_index]))
        except:
            print(curr_index, type(curr_index))
            self.data_h5.close()
            self.data_h5 = h5py.File(self.root, mode='r')
            im = PIL.Image.open(io.BytesIO(self.data_h5['x'][curr_index]))
        '''

        # print(curr_index, 'done')
        # convert gray to rgb
        print(im)
        if len(list(im.split())) == 1:
            im = im.convert('RGB')
        if self.transform is not None:
            im = self.transform(im)
        return im, self.ys[index], index

    def get_label(self, index):
        return self.ys[index]

    def set_subset(self, I):
        self.ys = [self.ys[i] for i in I]
        self.I = [self.I[i] for i in I]
        self.im_paths = [self.im_paths[i] for i in I]


class BaseDatasetMod(torch.utils.data.Dataset):
    def __init__(self, root, source, classes, transform=None):
        self.classes = classes
        self.root = root
        self.transform = transform
        self.ys, self.im_paths, self.I = [], [], []

        if not os.path.exists(root):
            print('copying file from source to root')
            print('from:', source)
            print('to:', root)
            c_time = time.time()

            copy_tree(source, root)

            elapsed = time.time() - c_time
            print('done copying file: %.2fs', elapsed)

    def nb_classes(self):
        # print(self.classes)
        # print(len(set(self.ys)), len(set(self.classes)))
        # print(type(self.ys))
        # print(len(set(self.ys) & set(self.classes)))
        assert set(self.ys) == set(self.classes)
        return len(self.classes)

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, index):
        im = PIL.Image.open(self.im_paths[index])
        # convert gray to rgb
        try:
            if len(list(im.split())) == 1 : im = im.convert('RGB')
            if self.transform is not None:
                im = self.transform(im)
        except:
            print(self.im_paths[index])
        return im, self.ys[index], index

    def get_label(self, index):
        return self.ys[index]

    def set_subset(self, I):
        self.ys = [self.ys[i] for i in I]
        self.I = [self.I[i] for i in I]
        self.im_paths = [self.im_paths[i] for i in I]


class BaseDatasetMem(torch.utils.data.Dataset):
    def __init__(self, classes, transform=None):
        self.classes = classes
        self.transform = transform
        self.ys, self.im_paths, self.I = [], [], []

    def nb_classes(self):
        assert set(self.ys) == set(self.classes)
        return len(self.classes)

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, index):
        im = PIL.Image.fromarray(self.data[index].numpy())
        # print(curr_index, 'done')
        # convert gray to rgb
        if len(list(im.split())) == 1: im = im.convert('RGB')
        if self.transform is not None:
            im = self.transform(im)
        return im, self.ys[index], index

    def get_label(self, index):
        return self.ys[index]

    def set_subset(self, I):
        self.ys = [self.ys[i] for i in I]
        self.I = [self.I[i] for i in I]
        self.im_paths = [self.im_paths[i] for i in I]


class BaseDataset_hdf5_alt(torch.utils.data.Dataset):
    def __init__(self, root, source, classes, transform=None, prefix=''):
        self.classes = classes
        self.root = root
        self.transform = transform
        self.ys, self.im_paths, self.I = [], [], []
        self.data_h5 = None
        self.prefix = prefix

        # making sure it is not an old copy or broken copy
        if not os.path.exists(root):
            if not os.path.exists(os.path.dirname(root)):
                os.makedirs(os.path.dirname(root))

            print('copying file from source to root')
            print('from:', source)
            print('to:', root)
            c_time = time.time()
            copyfile(source, root)
            elapsed = time.time() - c_time
            print('done copying file: %.2fs' % elapsed)

    def nb_classes(self):
        assert set(self.ys) == set(self.classes)
        return len(self.classes)

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, index):
        # im = PIL.Image.open(self.im_paths[index])
        # if self.data_h5 is None:
        self.data_h5 = h5py.File(self.root, mode='r')

        curr_index = self.I[index]
        im = PIL.Image.open(io.BytesIO(self.data_h5[self.prefix + 'x'][curr_index]))
        path = self.data_h5[self.prefix + 'path'][curr_index]

        self.data_h5.close()
        ''' 
        try:
            im = PIL.Image.open(io.BytesIO(self.data_h5['x'][curr_index]))
        except:
            print(curr_index, type(curr_index))
            self.data_h5.close()
            self.data_h5 = h5py.File(self.root, mode='r')
            im = PIL.Image.open(io.BytesIO(self.data_h5['x'][curr_index]))
        '''

        # print(curr_index, 'done')
        # convert gray to rgb
        if len(list(im.split())) == 1: im = im.convert('RGB')
        if self.transform is not None:
            im = self.transform(im)

        return im, self.ys[index], index, path.decode()

    def get_label(self, index):
        return self.ys[index]

    def set_subset(self, I):
        self.ys = [self.ys[i] for i in I]
        self.I = [self.I[i] for i in I]
        self.im_paths = [self.im_paths[i] for i in I]


class BaseDataset_hdf5_bb(torch.utils.data.Dataset):
    def __init__(self, root, source, classes, transform=None, prefix=''):
        self.classes = classes
        self.root = root
        self.transform = transform
        self.ys, self.im_paths, self.I = [], [], []
        self.data_h5 = None
        self.prefix = prefix

        # making sure it is not an old copy or broken copy
        if not os.path.exists(root):
            if not os.path.exists(os.path.dirname(root)):
                os.makedirs(os.path.dirname(root))

            print('copying file from source to root')
            print('from:', source)
            print('to:', root)
            c_time = time.time()
            copyfile(source, root)
            elapsed = time.time() - c_time
            print('done copying file: %.2fs' % elapsed)

    def nb_classes(self):
        assert set(self.ys) == set(self.classes)
        return len(self.classes)

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, index):
        # im = PIL.Image.open(self.im_paths[index])
        # if self.data_h5 is None:
        self.data_h5 = h5py.File(self.root, mode='r')

        curr_index = self.I[index]
        im = PIL.Image.open(io.BytesIO(self.data_h5[self.prefix + 'x'][curr_index]))
        path = self.data_h5[self.prefix + 'path'][curr_index]

        x1 = self.data_h5[self.prefix + 'x1'][curr_index]
        x2 = self.data_h5[self.prefix + 'x2'][curr_index]
        y1 = self.data_h5[self.prefix + 'y1'][curr_index]
        y2 = self.data_h5[self.prefix + 'y2'][curr_index]

        im = im.crop((x1, y1, x2, y2))

        self.data_h5.close()
        ''' 
        try:
            im = PIL.Image.open(io.BytesIO(self.data_h5['x'][curr_index]))
        except:
            print(curr_index, type(curr_index))
            self.data_h5.close()
            self.data_h5 = h5py.File(self.root, mode='r')
            im = PIL.Image.open(io.BytesIO(self.data_h5['x'][curr_index]))
        '''

        # print(curr_index, 'done')
        # convert gray to rgb
        if len(list(im.split())) == 1: im = im.convert('RGB')
        if self.transform is not None:
            im = self.transform(im)

        return im, self.ys[index], index  # , path.decode()

    def get_label(self, index):
        return self.ys[index]

    def set_subset(self, I):
        self.ys = [self.ys[i] for i in I]
        self.I = [self.I[i] for i in I]
        self.im_paths = [self.im_paths[i] for i in I]
