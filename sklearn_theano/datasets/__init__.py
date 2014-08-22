"""This module deals with loading datasets."""
from .asirra import fetch_asirra
from .base import get_dataset_dir
from .base import download
from .base import load_sample_images
from .generators import fetch_mnist_generated

__all__ = ['fetch_asirra',
           'fetch_mnist_generated',
           'load_sample_images',
           'get_dataset_dir',
           'download']
