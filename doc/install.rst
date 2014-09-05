.. _installation-instructions:

=======================
Installing sklearn-theano
=======================
.. note::

    If you wish to contribute to the project, it's recommended you
    :ref:`install the latest development version <install_development_version>`.


.. _install_development_version:

Installing
==========

Sklearn-theano requires:

- Python (>= 2.6 or >= 3.3),
- Numpy
- Scipy
- Theano (>= 0.6)
- scikit-learn
- pillow

Retrieving the codebase should be straightforward:

``git clone https://github.com/sklearn-theano/sklearn-theano``

Next, go into the directory where the clone was placed
(generally sklearn-theano) and run:

``python setup.py develop``

If for some reason you want to install as a package instead of development:

``python setup.py install``

will work too.
