sklearn-theano
==============

This repository contains experiments with scikit-learn compatible estimators,
transformers, and datasets which use Theano internally.

Setup
============

``python setup.py develop``

When using the examples, there will be some fairly large downloads (~1GB) to
get weights, sample datasets, and other useful tools. The default directory for
this is ``$HOME/sklearn_theano_data``.

The key packages required are:
* numpy
* scipy
* theano
* scikit-learn

and a soft dependency on matplotlib for the examples. 

Documentation is sparse but we are working to improve unclear modules. Feel free
to raise issues with any problems found!
