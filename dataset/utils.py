from __future__ import print_function
from __future__ import division

import torchvision
from torchvision import transforms
import PIL.Image
import torch
#from torch._six import int_classes as _int_classes
import numpy as np
import numbers


def std_per_channel(images):
    images = torch.stack(images, dim=0)
    return images.view(3, -1).std(dim=1)


def mean_per_channel(images):
    images = torch.stack(images, dim=0)
    return images.view(3, -1).mean(dim=1)


class Identity:  # used for skipping transforms
    def __call__(self, im):
        return im


class RGBToBGR:
    def __call__(self, im):
        assert im.mode == 'RGB'
        r, g, b = [im.getchannel(i) for i in range(3)]
        # RGB mode also for BGR, `3x8-bit pixels, true color`, see PIL doc
        im = PIL.Image.merge('RGB', [b, g, r])
        return im


class ScaleIntensities:
    def __init__(self, in_range, out_range):
        """ Scales intensities. For example [-1, 1] -> [0, 255]."""
        self.in_range = in_range
        self.out_range = out_range

    def __oldcall__(self, tensor):
        tensor.mul_(255)
        return tensor

    def __call__(self, tensor):
        tensor = (
                         tensor - self.in_range[0]
                 ) / (
                         self.in_range[1] - self.in_range[0]
                 ) * (
                         self.out_range[1] - self.out_range[0]
                 ) + self.out_range[0]
        return tensor


def make_transform(sz_resize=256, sz_crop=227, mean=None,
                   std=None, rgb_to_bgr=True, is_train=True,
                   intensity_scale=None, rotate=0, jitter=0, hue=0):
    if std is None:
        std = [1, 1, 1]
    if mean is None:
        mean = [104, 117, 128]
    return transforms.Compose([
        RGBToBGR() if rgb_to_bgr else Identity(),
        transforms.RandomRotation(rotate) if is_train else Identity(),
        transforms.RandomResizedCrop(sz_crop) if is_train else Identity(),
        transforms.Resize(sz_resize) if not is_train else Identity(),
        transforms.CenterCrop(sz_crop) if not is_train else Identity(),
        transforms.RandomHorizontalFlip() if is_train else Identity(),
        #transforms.ColorJitter(brightness=jitter, contrast=jitter, saturation=jitter, hue=hue) if is_train else Identity(),
        
        #### New Added ####
        transforms.RandomPerspective() if is_train else Identity(),
        transforms.RandomEqualize() if is_train else Identity(),
        transforms.RandomVerticalFlip() if is_train else Identity(),
        transforms.RandomAdjustSharpness(sharpness_factor = 2) if is_train else Identity(),
        transforms.RandomAutocontrast() if is_train else Identity(),
        transforms.RandomErasing() if is_train else Identity(),
        transforms.RandomAffine(degrees=(-60, 60), translate=(0.1, 0.3), scale=(0.5, 0.75)) if is_train else Identity(),
        #### New Added ####

        transforms.ToTensor(),
        ScaleIntensities(
            *intensity_scale) if intensity_scale is not None else Identity(),
        transforms.Normalize(
            mean=mean,
            std=std,
        )
    ])


class BatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, batch_size, drop_last, dataset, sel_class):
        int_classes = int()
        if not isinstance(batch_size, int_classes) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.dataset = dataset

        self.dataset.pop_class_list()
        self.dataset.sel_class = sel_class
        self.dataset.resel_random_classes()

    def __iter__(self):
        batch = []
        for idx in range(len(self.dataset)):
            rand_class = self.dataset.random_classes[idx % self.dataset.sel_class]
            class_list = self.dataset.class_list[rand_class]
            idx = self.dataset.class_list[rand_class][torch.randperm(len(class_list))[0]]
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
                self.dataset.resel_random_classes()
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


class RandomBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, labels, batch_size, drop_last, sel_class, nb_gradcum=1):
        int_classes = int()
        if not isinstance(batch_size, int_classes) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.nb_gradcum = nb_gradcum

        self.labels = labels
        self.class_list = []
        for ix in range(len(set(labels))): self.class_list.append([])
        for ix in range(len(labels)):
            self.class_list[labels[ix]].append(ix)
        self.sel_class = sel_class
        self.random_classes = torch.randperm(len(self.class_list))[:self.sel_class]

    def __iter__(self):
        batch = []
        bc = 0
        # for idx in range(len(self.dataset)):
        for idx in range(len(self.labels)):
            # rand_class = self.dataset.random_classes[idx % self.dataset.sel_class]
            rand_class = self.random_classes[torch.randperm(len(self.random_classes))[0]]
            class_list = self.class_list[rand_class]
            idx = self.class_list[rand_class][torch.randperm(len(class_list))[0]]
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
                bc += 1
                # self.dataset.resel_random_classes()
            if bc == self.nb_gradcum:
                bc = 0
                # self.dataset.resel_random_classes()
                self.random_classes = torch.randperm(len(self.class_list))[:self.sel_class]
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.labels) // self.batch_size
        else:
            return (len(self.labels) + self.batch_size - 1) // self.batch_size


class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size
