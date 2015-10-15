"""Base datasets functionality."""
# Authors: Kyle Kastner
#          Michael Eickenberg
# License: BSD 3 Clause

import os
import numpy as np
from PIL import Image
from sklearn.datasets.base import Bunch
try:
    import urllib.request as urllib  # for backwards compatibility
except ImportError:
    import urllib2 as urllib


def get_dataset_dir(dataset_name, data_dir=None, folder=None, create_dir=True):
    if not data_dir:
        data_dir = os.getenv("SKLEARN_THEANO_DATA", os.path.join(
            os.path.expanduser("~"), "sklearn_theano_data"))
    if folder is None:
        data_dir = os.path.join(data_dir, dataset_name)
    else:
        data_dir = os.path.join(data_dir, folder)
    if not os.path.exists(data_dir) and create_dir:
        os.makedirs(data_dir)
    return data_dir


def download(url, server_fname, local_fname=None, progress_update_percentage=5,
             bypass_certificate_check=False):
    """
    An internet download utility modified from
    http://stackoverflow.com/questions/22676/
    how-do-i-download-a-file-over-http-using-python/22776#22776
    """
    if bypass_certificate_check:
        import ssl
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        u = urllib.urlopen(url, context=ctx)
    else:
        u = urllib.urlopen(url)
    if local_fname is None:
        local_fname = server_fname
    full_path = local_fname
    meta = u.info()
    with open(full_path, 'wb') as f:
        try:
            file_size = int(meta.get("Content-Length"))
        except TypeError:
            print("WARNING: Cannot get file size, displaying bytes instead!")
            file_size = 100
        print("Downloading: %s Bytes: %s" % (server_fname, file_size))
        file_size_dl = 0
        block_sz = int(1E7)
        p = 0
        while True:
            buffer = u.read(block_sz)
            if not buffer:
                break
            file_size_dl += len(buffer)
            f.write(buffer)
            if (file_size_dl * 100. / file_size) > p:
                status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl *
                                               100. / file_size)
                print(status)
                p += progress_update_percentage


def load_images(filenames):
    """Load images for image manipulation.

    Parameters
    ----------
    filenames : iterable
         Iterable of filename paths as strings

    Returns
    -------
    data : Bunch
        Dictionary-like object with the following attributes :
        'images', the sample images, 'filenames', the file
        names for the images
    """
    # Load image data for each image in the source folder.
    images = [np.array(Image.open(filename, 'r')) for filename in filenames]

    return Bunch(images=images,
                 filenames=filenames)


def load_sample_images():
    """Load sample images for image manipulation.
    Loads ``sloth``, ``sloth_closeup``, ``cat_and_dog``.

    Returns
    -------
    data : Bunch
        Dictionary-like object with the following attributes :
        'images', the sample images, 'filenames', the file
        names for the images, and 'DESCR'
        the full description of the dataset.
    """
    module_path = os.path.join(os.path.dirname(__file__), "images")
    with open(os.path.join(module_path, 'README.txt')) as f:
        descr = f.read()
    filenames = [os.path.join(module_path, filename)
                 for filename in os.listdir(module_path)
                 if filename.endswith(".jpg")]
    # Load image data for each image in the source folder.
    images = [np.array(Image.open(filename, 'r')) for filename in filenames]

    return Bunch(images=images,
                 filenames=filenames,
                 DESCR=descr)


def load_sample_image(image_name):
    """Load the numpy array of a single sample image

    Parameters
    -----------
    image_name: {`sloth.jpg`, `sloth_closeup.jpg`, `cat_and_dog.jpg`}
        The name of the sample image loaded

    Returns
    -------
    img: 3D array
        The image as a numpy array: height x width x color

    """
    images = load_sample_images()
    index = None
    for i, filename in enumerate(images.filenames):
        if filename.endswith(image_name):
            index = i
            break
    if index is None:
        raise AttributeError("Cannot find sample image: %s" % image_name)
    return images.images[index]
