.. _feature_extraction:

============================
Feature extraction utilities
============================

.. currentmodule:: sklearn_theano.feature_extraction
This package also features helpers to fetch larger datasets and parameters
commonly used by the machine learning community to benchmark algorithm on data
that comes from the 'real world'.

.. _sample_images:

Overfeat
========

sklearn-theano wraps a variety of trained neural networks for use as
"black-box" transforms and classifiers. One of these networks in is known
as Overfeat, seen in the publication:

P. Sermanet, D. Eigen, X. Zhang, M. Mathieu, R. Fergus, Y. LeCun. *OverFeat: Integrated Recognition, Localization, and Detection using Convolutional Networks*, International Conference on Learning Representations (ICLR 2014), April 2014. 

Various layers of this neural network can be used for
transformation, classification, and localization.

.. autosummary::

   :toctree: ../modules/generated/
   :template: class.rst

   OverfeatTransformer
   OverfeatClassifier
   OverfeatLocalizer

.. image:: ../auto_examples/images/plot_multiple_localization_001.png
   :target: ../auto_examples/plot_multiple_localization.html
   :scale: 30
.. topic:: Examples:


    * :ref:`plot_multiple_localization.py`

GoogLeNet
========

Another state of the art neural network from Google is known as GoogLeNet:

C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, A. Rabinovich. *Going Deeper with Convolutions*, arXiv:1409.4842, September 2014.

Various layers of this neural network can be used for
transformation and classification.

.. autosummary::

    :toctree: ../modules/generated/
    :template: class.rst

    GoogLeNetTransformer
    GoogLeNetClassifier

.. image:: ../auto_examples/images/plot_comparison_001.png
   :target: ../auto_examples/plot_comparison.html
   :scale: 30
.. topic:: Examples:


    * :ref:`plot_comparison.py`
