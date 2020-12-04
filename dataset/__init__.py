import os

from .cars import Cars, Cars_hdf5
from .cub import CUBirds, CUBirds_hdf5, CUBirds_class, CUBirds_hdf5_alt, CUBirds_hdf5_bb
from .sop import SOProducts, SOProducts_hdf5
from .inshop import InShop, InShop_hdf5
from . import utils
from .shoes import Shoes

if os.name == 'posix':
    import resource  # pylint: disable=import-error

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    print(rlimit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (rlimit[1], rlimit[1]))

_type = {
    'cars': Cars,
    'cars_h5': Cars_hdf5,
    'cub': CUBirds,
    'cub_h5': CUBirds_hdf5,
    'cub_class': CUBirds_class,
    'sop': SOProducts,
    'shoes': Shoes,
    'sop_h5': SOProducts_hdf5,
    'sop_h5_mod': SOProducts_hdf5,
    'inshop': InShop,
    'inshop_h5': InShop_hdf5,
}


def load(name, root, source, classes, transform=None):
    return _type[name](root=root, source=source, classes=classes, transform=transform)


def load_inshop(name, root, source, classes, transform=None, dset_type='train'):
    return _type[name](root=root, source=source, classes=classes, transform=transform, dset_type=dset_type)
