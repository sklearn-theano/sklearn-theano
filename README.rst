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

License
=======

The License for sklearn-theano is 3-clause BSD. See the `LICENSE` file in the 
top level of the repository https://github.com/sklearn-theano/sklearn-theano/blob/master/LICENSE

This project provides downloaders for models that are distributed under their own terms, namely:
    * The OverFeat model http://cilvr.nyu.edu/doku.php?id=code:start
    * The BVLC Caffe GoogLeNet model http://caffe.berkeleyvision.org/

The model specification for the BVLC Caffe GoogLeNet model are taken from a
protocol buffer file, https://raw.githubusercontent.com/BVLC/caffe/master/src/caffe/proto/caffe.proto which is distributed under the same licence as the Caffe code (2-clause BSD).
