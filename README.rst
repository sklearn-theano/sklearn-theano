sklearn-theano
==============

This repository contains experiments with scikit-learn compatible estimators,
transformers, and datasets which use Theano internally.

Important Links
===============
- Official source code repo: https://github.com/sklearn-theano/sklearn-theano
- HTML Documentation: http://sklearn-theano.github.io

Setup
=====

The HTML docmentation linked above has a variety of resources for installation
and example use. To get started immediately, clone this repo then do:

``python setup.py develop``

When using the examples, there will be some fairly large downloads (~1GB) to
get weights, sample datasets, and other useful tools. The default directory for
this is ``$HOME/sklearn_theano_data``.

The key packages required are:
    * numpy
    * scipy
    * theano
    * scikit-learn
    * pillow

and a soft dependency on matplotlib for the examples. 

Documentation is sparse but we are working to improve unclear modules. Feel
free to raise issues on
`GitHub <https://github.com/sklearn-theano/sklearn-theano>`_
with any problems found!
