.. _contributing:

============
Contributing
============

This project is a community effort, and everyone is welcome to
contribute.

The project is hosted on http://github.com/sklearn-theano/sklearn-theano

Note
----

A large part of our guidelines come directly from `scikit-learn <http://scikit-learn.org/stable/developers/index.html>`_.
For more detail, refer to the documentation there.


Submitting a bug report
=======================

In case you experience issues using this package, do not hesitate to submit a
ticket to the
`Bug Tracker <http://github.com/sklearn-theano/sklearn-theano/issues>`_. You are
also welcome to post feature requests or links to pull requests.


.. _git_repo:

Retrieving the latest code
==========================

We use `Git <http://git-scm.com/>`_ for version control and
`GitHub <http://github.com/>`_ for hosting our main repository.

You can check out the latest sources with the command::

    git clone git://github.com/sklearn-theano/sklearn-theano.git

Contributing code
=================

.. note::

  To avoid duplicating work, it is highly advised that you contact the
  developers on GitHub before starting work on a non-trivial feature.
Opening a pull request with a [WIP] prefix usually serves to inform the
other developers that there is work happening, and any discussion
can happen there. For example, a pull request with the title
[WIP]World's best classifier would let us know you are implementing
the world's best classifier.

See the section below for more details on opening pull requests and
contributing to the project.

How to contribute
-----------------

The preferred way to contribute to sklearn-theano is to fork the `main
repository <http://github.com/sklearn-theano/sklearn-theano/>`__ on GitHub,
then submit a "pull request" (PR):

 1. `Create an account <https://github.com/signup/free>`_ on
    GitHub if you do not already have one.

 2. Fork the `project repository
    <http://github.com/sklearn-theano/sklearn-theano>`__: click on the 'Fork'
    button near the top of the page. This creates a copy of the code under your
    account on the GitHub server.

 3. Clone this copy to your local disk::

        $ git clone git@github.com:YourLogin/sklearn-theano.git

 4. Create a branch to hold your changes::

        $ git checkout -b my-feature

    and start making changes. Never work in the ``master`` branch!

 5. Work on this copy, on your computer, using Git to do the version
    control. When you're done editing, do::

        $ git add modified_files
        $ git commit

    to record your changes in Git, then push them to GitHub with::

        $ git push -u origin my-feature

Finally, go to the web page of the your fork of the scikit-learn repo,
and click 'Pull request' to send your changes to the maintainers for review.
request. This will send an email to the committers, but might also send an
email to the mailing list in order to get more visibility.

.. note::

  In the above setup, your ``origin`` remote repository points to
  YourLogin/sklearn-theano.git. If you wish to fetch/merge from the main
  repository instead of your forked one, you will need to add another remote
  to use instead of ``origin``. If we choose the name ``upstream`` for it, the
  command will be::

        $ git remote add upstream https://github.com/sklearn-theano/sklearn-theano.git

(If any of the above seems like magic to you, then look up the
`Git documentation <http://git-scm.com/documentation>`_ on the web.)

It is recommended to check that your contribution complies with the following
rules before submitting a pull request:

    * Follow the `coding-guidelines`_ (see below).

    * All public methods should have informative docstrings with sample
      usage presented as doctests when appropriate.

    * All other tests pass when everything is rebuilt from scratch. On
      Unix-like systems, check with (from the toplevel source folder)::

        $ make

    * When adding additional functionality, provide at least one example script
      in the ``examples/`` folder. Have a look at other examples for reference.
      Examples should demonstrate why the new functionality is useful in
      practice and, if possible, compare it to other methods available in
      sklearn-theano.

    * At least one paragraph of narrative documentation with links to
      references in the literature (with PDF links when possible) and
      the example. 

You can also check for common programming errors with the following tools:

    * Code with a good unittest coverage (at least 90%, better 100%), check
      with::

        $ pip install nose coverage
        $ nosetests --with-coverage path/to/tests_for_package

      see also :ref:`testing_coverage`

    * No pyflakes warnings, check with::

        $ pip install pyflakes
        $ pyflakes path/to/module.py

    * No PEP8 warnings, check with::

        $ pip install pep8
        $ pep8 path/to/module.py

    * AutoPEP8 can help you fix some of the easy redundant errors::

        $ pip install autopep8
        $ autopep8 path/to/pep8.py

Bonus points for contributions that include a performance analysis with
a benchmark script and profiling output (please report on the mailing
list or on the GitHub wiki).

.. note::

  The current state of the sklearn-theano code base is not compliant with
  all of those guidelines, but we expect that enforcing those constraints
  on all contributions will get the overall code base quality in the
  right direction.

.. note::

   For two very well documented and more detailed guides on development
   workflow, please pay a visit to the `Scipy Development Workflow
   <http://docs.scipy.org/doc/numpy/dev/gitwash/development_workflow.html>`_ -
   and the `Astropy Workflow for Developers
   <http://astropy.readthedocs.org/en/latest/development/workflow/development_workflow.html>`_
   sections.

.. _contribute_documentation:

Documentation
-------------

We are glad to accept any sort of documentation: function docstrings,
reStructuredText documents (like this one), tutorials, etc. reStructuredText
documents live in the source code repository under the doc/ directory.

You can edit the documentation using any text editor, and then generate the
HTML output by typing ``make html`` from the doc/ directory. Alternatively,
``make html-noplot`` can be used to quickly generate the documentation without
the example gallery. The resulting HTML files will be placed in _build/html/
and are viewable in a web browser. See the README file in the doc/ directory
for more information.

For building the documentation, you will need `sphinx
<http://sphinx.pocoo.org/>`_,
`matplotlib <http://matplotlib.sourceforge.net/>`_ and
`pillow <http://pillow.readthedocs.org/en/latest/>`_.

**When you are writing documentation**, it is important to keep a good
compromise between mathematical and algorithmic details, and give
intuition to the reader on what the algorithm does.

Any math and equations, followed by references,
can be added to further the documentation. Not starting the
documentation with the maths makes it more friendly towards
users that are just interested in what the feature will do, as
opposed to how it works "under the hood".


.. warning:: **Sphinx version**

   While we do our best to have the documentation build under as many
   version of Sphinx as possible, the different versions tend to behave
   slightly differently. To get the best results, you should use version
   1.0.

Issue Tracker Tags
------------------
All issues and pull requests on the
`Github issue tracker <https://github.com/scikit-learn/scikit-learn/issues>`_
should have (at least) one of the following tags:

:Bug / Crash:
    Something is happening that clearly shouldn't happen.
    Wrong results as well as unexpected errors from estimators go here.

:Cleanup / Enhancement:
    Improving performance, usability, consistency.

:Documentation:
    Missing, incorrect or sub-standard documentations and examples.

:New Feature:
    Feature requests and pull requests implementing a new feature.

There are two other tags to help new contributors:

:Easy:
    This issue can be tackled by anyone, no experience needed.
    Ask for help if the formulation is unclear.

:Moderate:
    Might need some knowledge of machine learning or the package,
    but is still approachable for someone new to the project.


Other ways to contribute
========================

Code is not the only way to contribute to sklearn-theano. For instance,
documentation is also a very important part of the project and often
doesn't get as much attention as it deserves. If you find a typo in
the documentation, or have made improvements, do not hesitate to send
an email to the mailing list or submit a GitHub pull request. Full
documentation can be found under the doc/ directory.

It also helps us if you spread the word: reference the project from your blog
and articles, link to it from your website, or simply say "I use it":

.. _coding-guidelines:

Coding guidelines
=================

The following are some guidelines on how new code should be written. Of
course, there are special cases and there will be exceptions to these
rules. However, following these rules when submitting new code makes
the review easier so new code can be integrated in less time.

Uniformly formatted code makes it easier to share code ownership. The
scikit-learn project tries to closely follow the official Python guidelines
detailed in `PEP8 <http://www.python.org/dev/peps/pep-0008/>`_ that
detail how code should be formatted and indented. Please read it and
follow it.

In addition, we add the following guidelines:

    * Use underscores to separate words in non class names: ``n_samples``
      rather than ``nsamples``.

    * Avoid multiple statements on one line. Prefer a line return after
      a control flow statement (``if``/``for``).

    * Use relative imports for references inside scikit-learn.

    * Unit tests are an exception to the previous rule;
      they should use absolute imports, exactly as client code would.
      A corollary is that, if ``sklearn_theano.foo`` exports a class or function
      that is implemented in ``sklearn_theano.foo.bar.baz``,
      the test should import it from ``sklearn_theano.foo``.

    * **Please don't use ``import *`` in any case**. It is considered harmful
      by the `official Python recommendations
      <http://docs.python.org/howto/doanddont.html#from-module-import>`_.
      It makes the code harder to read as the origin of symbols is no
      longer explicitly referenced, but most important, it prevents
      using a static analysis tool like `pyflakes
      <http://www.divmod.org/trac/wiki/DivmodPyflakes>`_ to automatically
      find bugs in sklearn-theano.

    * Use the `numpy docstring standard
      <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_
      in all your docstrings.


A good example of code that we like can be found `here
<https://svn.enthought.com/enthought/browser/sandbox/docs/coding_standard.py>`_.

Additional Help
---------------
See the `scikit-learn developer documentation <http://scikit-learn.org/stable/developers/index.html>`_
for more details. In general, we try to follow the scikit-learn guidelines as
closely as possible.

Working notes
-------------

For unresolved issues, TODOs, and remarks on ongoing work, developers are
advised to maintain notes on the `GitHub wiki
<https://github.com/sklearn-theano/sklearn-theano/wiki>`__.
