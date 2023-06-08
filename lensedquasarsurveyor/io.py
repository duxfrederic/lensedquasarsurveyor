"""
At some point we'll need to save stuff to disk for later use. We'll use hdf5 when possible, seems more
stable long term than pickle, and forces us to be a bit tidy.
"""
import h5py
import numpy as np


def save_dict_to_hdf5(filename, dic):
    with h5py.File(filename, 'w') as h5file:
        recursively_save_dict_contents_to_group(h5file, '/', dic)


def recursively_save_dict_contents_to_group(h5file, path, dic):
    # argument type checking
    if not isinstance(dic, dict):
        raise ValueError("must provide a dictionary")

    if not isinstance(path, str):
        raise ValueError("path must be a string")
    if not isinstance(h5file, h5py._hl.files.File):
        raise ValueError("must be an open h5py file")
    # save items to the hdf5 group
    for key, item in dic.items():
        if isinstance(item, (np.ndarray, np.generic)):
            h5file[path + key] = item
        elif isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
        else:
            raise ValueError('Cannot save %s type'%type(item))


def load_dict_from_hdf5(filename):
    with h5py.File(filename, 'r') as h5file:
        return recursively_load_dict_contents_from_group(h5file, '/')


def recursively_load_dict_contents_from_group(h5file, path):
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item[...]  # changed from 'item.value'
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans


def update_hdf5(filename, path, data):
    """
    Update the hdf5 at `filename`, under the internal path `path`, with the provided data.

    Exemple:
    PSF = np.ones((64,64))
    update_hdf5('yourfile.hdf5', 'band1/0/PSF', PSF)
    provided that the path band1/0 already exists.

    yeah I know, long ass docstring, but I never remember how to handle hdf5 files.

    :param filename: path or string, path to our hdf5 file.
    :param path: path inside the hdf5 path to update
    :param data: the data (numpy array) to be saved under this path.
    :return: None

    """
    with h5py.File(filename, 'a') as f:
        if path in f:
            del f[path]
        f[path] = data
