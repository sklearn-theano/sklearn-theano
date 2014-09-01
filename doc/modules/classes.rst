=========
Reference
=========

This is the class and function reference of scikit-learn. Please refer to
the :ref:`full user guide <user_guide>` for further details, as the class and
function raw specifications may not be enough to give full guidelines on their
uses.


.. _base_ref:

:mod:`sklearn_theano.base`: Base classes and utility functions
=======================================================

.. automodule:: sklearn_theano.base
    :no-members:
    :no-inherited-members:

Base classes
------------
.. currentmodule:: sklearn_theano

.. autosummary::
   :toctree: generated/
   :template: class.rst

   base.Feedforward
   base.Convolution
   base.PassThrough
   base.Standardize
   base.MaxPool

Functions
---------
.. currentmodule:: sklearn_theano

.. autosummary::
   :toctree: generated/
   :template: function.rst

   base.fuse


.. _datasets_ref:

:mod:`sklearn_theano.datasets`: Datasets
=================================

.. automodule:: sklearn_theano.datasets
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`datasets` section for further details.

Loaders
-------

.. currentmodule:: sklearn_theano

.. autosummary::
   :toctree: generated/
   :template: function.rst

   datasets.fetch_asirra
   datasets.load_sample_images
   datasets.load_sample_image

Samples generator
-----------------

.. currentmodule:: sklearn_theano

.. autosummary::
   :toctree: generated/
   :template: function.rst

   datasets.fetch_mnist_generated


.. _feature_extraction_ref:

:mod:`sklearn_theano.feature_extraction`: Feature Extraction
=====================================================

.. automodule:: sklearn_theano.feature_extraction
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`feature_extraction` section for further details.

From images
-----------

.. automodule:: sklearn_theano.feature_extraction
   :no-members:
   :no-inherited-members:

.. currentmodule:: sklearn_theano

.. autosummary::
   :toctree: generated/
   :template: class.rst

   feature_extraction.OverfeatTransformer
   feature_extraction.OverfeatClassifier
   feature_extraction.OverfeatLocalizer
