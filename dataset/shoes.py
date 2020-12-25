from .base import *


class Shoes(BaseDatasetMod):
    nb_train_all = 18365
    nb_test_all = 18283

    def __init__(self, root, source, classes, transform=None):
        BaseDatasetMod.__init__(self, root, source, classes, transform)
        classes_train = range(0, 3786)
        classes_test = range(3786, 7572)

        if classes.start in classes_train:
            if classes.stop - 1 in classes_train:
                train = True

        if classes.start in classes_test:
            if classes.stop - 1 in classes_test:
                train = False

        with open(
                os.path.join(
                    root,
                    'Shoes_{}.txt'.format('train' if train else 'test')
                )
        ) as f:

            f.readline()
            index = 0
            nb_images = 0

            for (image_id, class_id, _, path) in map(str.split, f):
                nb_images += 1
                if int(class_id) - 1 in classes:
                    self.im_paths.append(os.path.join(root, path))
                    self.ys.append(int(class_id) - 1)
                    self.I += [index]
                    index += 1

            if train:
                assert nb_images == type(self).nb_train_all
            else:
                assert nb_images == type(self).nb_test_all
