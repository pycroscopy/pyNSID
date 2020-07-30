Guidelines for Contribution
============================

We would like to thank you and several others who have offered / are willing to contribute their code.
We are more than happy to add your code to this project.
Just as we strive to ensure that you get the best possible software from us, we ask that you do the same for others.
We do NOT ask that your code be as efficient as possible. Instead, we have some simpler and easier requests.
We have compiled a list of best practices below with links to additional information.
If you are confused or need more help, please feel free to `contact us <./contact.html>`_.

Before you begin
----------------
1. All functionality in pyNSID revolves around the `N-Dimensional Spectroscopic and Imaging Data (NSID) <../nsid.html>`_ model
   and its implementation into HDF5 files. Data is read from HDF5 files, processed, and written back to it.
   Therefore, it will be much easier to understand the rationale for certain practices in pyNSID once NSID is understood.
2. `Get in touch <./contact.html>`_ with us to be added to the weekly hackathon so we can coordinate development activities with you.

Structuring code
----------------

General guidelines
~~~~~~~~~~~~~~~~~~
* Encapsulate independent sections of your code into functions that can be used individually if required.
* Ensure that your code (functions) is well documented (`numpy format <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_)
  - expected inputs and outputs, purpose of functions
* Please avoid very short names for variables like ``i`` or ``k``. This makes it challenging to follow code, find and fix bugs.
* Ensure that your code works in python 2.7 and python 3.7
* Please consider using packages that are easy to install on Windows, Mac, and Linux.
  It is quite likely that packages included within Anaconda (has a comprehensive list packages for science and data analysis + visualization) can handle most needs.
  If this is not possible, try to use packages that are easy to to install (pip install).
  If even this is not possible, try to use packages that at least have conda installers.
* Follow best practices for `PEP8 compatibility <https://www.datacamp.com/community/tutorials/pep8-tutorial-python-code>`_.
  The easiest way to ensure compatibility is to set it up in your code editor.
  `PyCharm <https://blog.jetbrains.com/pycharm/2013/02/long-awaited-pep-8-checks-on-the-fly-improved-doctest-support-and-more-in-pycharm-2-7/>`_ does this by default.
  So, as long as PyCharm does not raise many warning, your code is beautiful!

pyNSID-specific guidelines
~~~~~~~~~~~~~~~~~~~~~~~~~~
* Engineering / science-agnostic tools fit better into pyNSID while scientific functionality go into pycroscopy.
* Please ensure that your code files fit into our package structure (``io``, ``processing``, ``viz``)
* Once you decide where your code will sit, please use relative import statements instead of absolute / external paths.
  For example, if you are contributing code for a new submodule within ``pyNSID.io``, you will need to turn your import statements and code from something like:

  .. code-block:: python

     import pyNSID as nsid
     ...
     nsid.hdf_utils.print_tree(hdf_file_handle)
     my_dataset = nsid.NSIDataset(my_hdf5_dataset)

  to:

  .. code-block:: python

     from .hdf_utils import print_tree
     from .nsi_data import NSIDataset
     ...
     print_tree(hdf_file_handle)
     my_dataset = NSIDataset(my_hdf5_dataset)

You can look at our code in our `GitHub project <https://github.com/pycroscopy/pyNSID>`_ to get an idea of how we organize, document, and submit our code.

Contributing code
-----------------
We recommend that you follow the steps below. Again, if you are ever need help, please contact us:

1. Learn ``git`` if you are not already familiar with it. See our `compilation of tutorials and guides <./external_guides.html>`_, especially `this one <https://github.com/pycroscopy/pyUSID/blob/master/docs/Using%20PyCharm%20to%20manage%20repository.pdf>`_.
2. Create a ``fork`` of pyNSID - this creates a separate copy of the entire pyNSID repository under your user ID.
   For more information see `instructions here <https://help.github.com/articles/fork-a-repo/>`_.
3. Once inside your own fork, you can either work directly off ``master`` or create a new branch.
4. Add / modify code
5. ``Commit`` your changes (equivalent to saving locally on your laptop). Do this regularly.
6. Repeat steps 4-5.
7. After you reach a certain milestone, ``push`` your commits to your ``remote branch``.
   This synchronizes your changes with the GitHub website and is similar to the Dropbox website /service making note of changes in your documents.
   To avoid losing work due to problems with your computer, consider ``pushing commits`` once at least every day / every few days.
8. Repeat steps 4-7 till you are ready to have your code added to the parent pyNSID repository.
   At this point, `create a pull request <https://help.github.com/articles/creating-a-pull-request-from-a-fork/>`_.
   Someone on the development team will review your ``pull request``. If any changes are req and then ``merge`` these changes to ``master``.

Writing tests
-------------
Software can become complicated very quickly through a complex interconnected web of dependencies, etc.
Adding or modifying code at one location may break some use case or code in a different location.
Unit tests are short functions that test to see if functions / classes respond in the expected way given some known inputs.
Unit tests are a good start for ensuring that you spend more time using code than fixing it. New functions / classes must be accompanied with unit tests.

Additionally, examples on how to use the new code must also be added so others are aware about how to use the code.
Fortunately, it is rather straightforward to `turn unit tests into examples <./unit_tests_to_examples.html>`_.
