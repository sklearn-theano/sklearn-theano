.. We want the previous/next button to work on the user guide pages and on the
   API. We have to put the doctree so that sphinx populates the
   rellinks. Yet, we don't want it to be displayed on the main page, hence we
   don't display through the css.

.. raw:: html

   <div class="no-display">

.. toctree::
    user_guide
    auto_examples/index
    presentations
    about
    documentation
    datasets/index
    modules/classes
    developers/index
    install


.. raw:: html

   </div>


.. This is were the real work starts.


.. raw:: html

    <!-- Block section -->
    <div class="container-index">
    <div class="container index-upper">
    <div class="row-fluid">

    <!-- Classification -->
    <div class="span4 box">
    <h2 >

:ref:`Classification <feature_extraction>`

.. raw:: html

    </h2>
    <blockquote>
    <p>Identifying to which set of categories a new observation belongs
    to.</p>
    <div class="box-links">
    <strong>Applications</strong>: Image recognition, object localization.</br>
    <strong>Algorithms</strong>:&nbsp;

:ref:`OverfeatClassifier<feature_extraction>`, :ref:`OverfeatLocalizer<feature_extraction>`, ...

.. raw:: html

    <small class="float-right box-example-links">

:ref:`Examples<general_examples>`

.. raw:: html

    </small>
    </div>
    </blockquote>
    </div>

    <!-- Preprocessing -->
    <div class="span4 box">
    <h2>

:ref:`Preprocessing<feature_extraction>`

.. raw:: html

    </h2>
    <blockquote>
    <p>Feature extraction and normalization.</p>
    <div class="box-links">
    <strong>Application</strong>: Transforming input data such as images for use with machine learning algorithms.</br>
    <strong>Modules</strong>:&nbsp;

:ref:`OverfeatTransformer<feature_extraction>`...

.. raw:: html

    <small class="float-right example-links">

:ref:`Examples<general_examples>`

.. raw:: html

    </small>
    </div>
    </blockquote>
    </div>
    <!-- Classification -->
    <div class="span4 box">
    <h2 >

:ref:`Datasets <datasets>`

.. raw:: html

    </h2>
    <blockquote>
    <p>Downloading, managing, and loading for datasets and network weights.
    </p>
    <div class="box-links">
    <strong>Applications</strong>: Automatic loading of image datasets and network weights.</br>
    <strong>Modules</strong>:&nbsp;

:ref:`Asirra<datasets>`, :ref:`Generative MNIST<datasets>`, ...

.. raw:: html

    <small class="float-right box-example-links">

:ref:`Examples<general_examples>`

.. raw:: html

    </small>
    </div>
    </blockquote>
    </div>

