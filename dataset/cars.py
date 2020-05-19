
from .base import *
import scipy.io
'''
class Cars(BaseDatasetMod):
    def __init__(self, root, source, classes, transform = None):
        BaseDatasetMod.__init__(self, root, source, classes, transform)
        annos_fn = 'cars_anno.pt'
        cars = torch.load(os.path.join(root, annos_fn))
        index = 0
        ys = cars['labels']
        im_paths = cars['filenames']

        for im_path, y in zip(im_paths, ys):
            y = y - 1 
            im_path = '0' + im_path
            if y in classes: # choose only specified classes
                self.im_paths.append(os.path.join(root, im_path))
                self.ys.append(y)
                self.I += [index]
                index += 1


'''

class Cars(BaseDatasetMod):
    def __init__(self, root, source, classes, transform = None):
        BaseDatasetMod.__init__(self, root, source, classes, transform)
        annos_fn = 'cars_annos.mat'
        cars = scipy.io.loadmat(os.path.join(root, annos_fn))
        ys = [int(a[5][0] - 1) for a in cars['annotations'][0]]
        im_paths = [a[0][0] for a in cars['annotations'][0]]
        index = 0
        for im_path, y in zip(im_paths, ys):
            if y in classes: # choose only specified classes
                self.im_paths.append(os.path.join(root, im_path))
                self.ys.append(y)
                self.I += [index]
                index += 1

class Cars_hdf5(BaseDataset_hdf5):
    def __init__(self, root, source, classes, transform = None):
        BaseDataset_hdf5.__init__(self, root, source, classes, transform)

        index = 0
        self.data_y = h5py.File(root, 'r')
        self.all_labels = torch.Tensor(self.data_y['y']).squeeze().long()
        self.data_y.close()
        self.data_y = None
        print(self.classes)
        for ix in range(len(self.all_labels)):
            if self.all_labels[ix] in self.classes:
                self.ys += [self.all_labels[ix].item()]
                self.I += [ix]
                index += 1

