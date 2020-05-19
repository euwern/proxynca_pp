from .base import *
import h5py
import torch

class CUBirds(BaseDatasetMod):
    def __init__(self, root, source, classes, transform = None):
        BaseDatasetMod.__init__(self, root, source, classes, transform)
        index = 0
        for i in torchvision.datasets.ImageFolder(root = 
                os.path.join(root, 'images')).imgs:
            # i[1]: label, i[0]: root
            y = i[1]
            # fn needed for removing non-images starting with `._`
            fn = os.path.split(i[0])[1]
            if y in self.classes and fn[:2] != '._':
                self.ys += [y]
                self.I += [index]
                self.im_paths.append(os.path.join(root, i[0]))
                index += 1

class CUBirds_hdf5(BaseDataset_hdf5):
    def __init__(self, root, source, classes, transform = None):
        BaseDataset_hdf5.__init__(self, root, source, classes, transform)

        index = 0
        self.data_y = h5py.File(root, 'r')
        self.all_labels = torch.Tensor(self.data_y['y']).squeeze().long()
        self.data_y.close()
        self.data_y = None
        for ix in range(len(self.all_labels)):
            if self.all_labels[ix] in self.classes:
                self.ys += [self.all_labels[ix].item()]
                self.I += [ix]
                index += 1

class CUBirds_hdf5_alt(BaseDataset_hdf5_alt):
    def __init__(self, root, source, classes, transform = None):
        BaseDataset_hdf5_alt.__init__(self, root, source, classes, transform)

        index = 0
        self.data_y = h5py.File(root, 'r')
        self.all_labels = torch.Tensor(self.data_y['y']).squeeze().long()
        self.data_y.close()
        self.data_y = None
        for ix in range(len(self.all_labels)):
            if self.all_labels[ix] in self.classes:
                self.ys += [self.all_labels[ix].item()]
                self.I += [ix]
                index += 1

class CUBirds_hdf5_bb(BaseDataset_hdf5_bb):
    def __init__(self, root, source, classes, transform = None):
        BaseDataset_hdf5_bb.__init__(self, root, source, classes, transform)

        index = 0
        self.data_y = h5py.File(root, 'r')
        self.all_labels = torch.Tensor(self.data_y['y']).squeeze().long()
        self.data_y.close()
        self.data_y = None
        for ix in range(len(self.all_labels)):
            if self.all_labels[ix] in self.classes:
                self.ys += [self.all_labels[ix].item()]
                self.I += [ix]
                index += 1

class CUBirds_class(BaseDatasetMod):
    def __init__(self, root, source, classes, transform = None, mode='train'):
        BaseDatasetMod.__init__(self, root, source, classes, transform)
        index = 0
        
        for i in torchvision.datasets.ImageFolder(root = 
                os.path.join(root, 'images')).imgs:
            # i[1]: label, i[0]: root
            y = i[1]
            # fn needed for removing non-images starting with `._`
            fn = os.path.split(i[0])[1]
            if y in self.classes and fn[:2] != '._':
                self.ys += [y]
                self.I += [index]
                self.im_paths.append(os.path.join(root, i[0]))
                index += 1

        cut_off = int(len(self.ys)*0.5)
        torch.manual_seed(1)
        rand_list = torch.randperm(len(self.ys)).tolist()
        
        ys = []
        I = []
        paths = []
        if mode == 'train':
            for ix in range(len(self.ys)):
                if ix < cut_off:
                    ys.append(self.ys[rand_list[ix]])
                    I.append(self.I[rand_list[ix]])
                    paths.append(self.im_paths[rand_list[ix]])
        else:
            for ix in range(len(self.ys)):
                if ix >= cut_off:
                    ys.append(self.ys[rand_list[ix]])
                    I.append(self.I[rand_list[ix]])
                    paths.append(self.im_paths[rand_list[ix]])
       
        self.ys = ys
        self.I = I
        self.im_paths = paths
