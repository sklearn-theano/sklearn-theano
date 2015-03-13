"""Dataset loading utilities for asirra dataset."""
# Authors: Kyle Kastner
# License: BSD 3 Clause

import os
import numpy as np
from PIL import Image
import tarfile
from glob import glob

from sklearn.externals.joblib import Memory
from sklearn.datasets.base import Bunch
from .base import download, get_dataset_dir


def check_fetch_asirra():
    url = "ftp://ftp.research.microsoft.com/pub/asirra/petimages.tar"
    partial_path = get_dataset_dir("asirra")
    full_path = os.path.join(partial_path, "petimages.tar")
    if not os.path.exists(partial_path):
        os.makedirs(partial_path)
    if not os.path.exists(full_path):
        download(url, full_path, progress_update_percentage=1)
    files_path = os.path.join(partial_path, "PetImages")
    if not os.path.exists(files_path):
        tar_obj = tarfile.TarFile(full_path, 'r')
        tar_obj.extractall(partial_path)
        tar_obj.close()
    return partial_path


def _fetch_asirra(partial_path, image_count=1000):
    files_path = os.path.join(partial_path, 'PetImages')
    cat_files_path = os.path.join(files_path, 'Cat', '*.jpg')
    dog_files_path = os.path.join(files_path, 'Dog', '*.jpg')

    def get_file_number(f):
        fname = os.path.split(f)[-1]
        return int(fname[:-4])

    cat_files = sorted(glob(cat_files_path), key=get_file_number)
    dog_files = sorted(glob(dog_files_path), key=get_file_number)
    X = np.zeros((image_count, 231, 231, 3), dtype='uint8')
    y = np.zeros(image_count)
    count = 0
    for f in cat_files:
        try:
            # Some of these files are not reading right...
            im = Image.open(f, 'r')
            X[count] = np.array(im.resize((231, 231)))
            y[count] = 0.
            count += 1
        except:
            continue
        if count >= image_count // 2:
            break

    for f in dog_files:
        try:
            # Some of these files are not reading right...
            im = Image.open(f, 'r')
            X[count] = np.array(im.resize((231, 231)))
            y[count] = 1.
            count += 1
        except:
            continue
        if count >= image_count:
            break
    return X, y


def fetch_asirra(image_count=1000):
    """

    Parameters
    ----------
    image_count : positive integer

    Returns
    -------
    data : Bunch
        Dictionary-like object with the following attributes :
        'images', the sample images, 'data', the flattened images,
        'target', the label for the image (0 for cat, 1 for dog),
        and 'DESCR' the full description of the dataset.
    """
    partial_path = check_fetch_asirra()
    m = Memory(cachedir=partial_path, compress=6, verbose=0)
    load_func = m.cache(_fetch_asirra)
    images, target = load_func(partial_path, image_count=image_count)
    return Bunch(data=images.reshape(len(images), -1),
                 images=images, target=target,
                 DESCR="Asirra cats and dogs dataset")
