.. _datasets:

=========================
Dataset loading utilities
=========================

.. currentmodule:: sklearn_theano.datasets
This package also features helpers to fetch larger datasets and parameters
commonly used by the machine learning community to benchmark algorithm on data
that comes from the 'real world'.

.. _sample_images:

Sample images
=============

sklearn-theano embeds sample JPEG images published under Creative
Commons license by their authors. These images can be useful to test
algorithms and pipelines for images and other multidimensional data.

.. autosummary::

   :toctree: ../modules/generated/
   :template: function.rst

   load_sample_images
   load_sample_image

.. image:: ../auto_examples/images/plot_single_localization_001.png
   :target: ../auto_examples/plot_single_localization.html
   :scale: 30
   :align: right


.. warning::

  The default coding of images is based on the ``uint8`` dtype to
  spare memory.  Often machine learning algorithms work best if the
  input is converted to a floating point representation first.  Also,
  if you plan to use ``pylab.imshow`` don't forget to scale to the range
  0 - 1 as done in the following example.

.. topic:: Examples:

    * :ref:`plot_single_localization.py`


.. _sample_generators:

Sample generators
=================

In addition, sklearn-theano includes various random sample generators that
can be used to build artificial datasets of controlled size.

.. image:: ../auto_examples/images/plot_mnist_generator_001.png
   :target: ../auto_examples/datasets/plot_mnist_generator.html
   :scale: 50
   :align: center

.. autosummary::

   :toctree: ../modules/generated/
   :template: function.rst

   fetch_mnist_generated

.. _larger_datasets:

Larger datasets
===============

sklearn-theano also includes downloaders for larger datasets that
can be used for something closer to "real world" testing.

.. image:: ../auto_examples/images/plot_asirra_dataset_001.png
   :target: ../auto_examples/datasets/plot_asirra_dataset.html
   :scale: 50
   :align: center

.. autosummary::

   :toctree: ../modules/generated/
   :template: function.rst

   fetch_asirra

