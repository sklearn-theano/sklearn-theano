"""This module deals with loading datasets."""
from .asirra import fetch_asirra
from .base import get_dataset_dir
from .base import download
from .generators import fetch_mnist_generated

__all__ = ['fetch_asirra',
           'fetch_mnist_generated',
           'get_dataset_dir',
           'download']
