======
pyNSID
======

**Python framework for storing N-dimensional scientific data**

What?
------
* The `N-Dimensional Spectroscopic and Imaging Data (NSID) model <../nsid.html>`_:

  * itself is a definition or specification for how to store data.
  * facilitates the representation of any N-dimensional data array regardless of their origin, modality, size, or dimensionality.
  * can be used for any N-dimensional data (scientific or otherwise), though originally designed for spectroscopic and imaging data.
  * simplifies downstream development of instrument- and modality- agnostic data processing and analysis algorithms.
  * cannot handle niche cases like spiral scans, compressed sensing, etc. given that these data do not have an N-dimensional form.
    However, our sister project - `pyUSID <../pyUSID/about.html>`_ was built to handle such complex scenarios
* pyNSID is a `python <http://www.python.org/>`_ package that currently provides all  **io** functionality:

  * * pyNSID is build on top of `h5py <https://docs.h5py.org/en/stable/>`_ a popular package for reading and manipulating hierarchical data file format (HDF5) files.
  * **io**: Primarily, it enables the storage and access of NSID in **hierarchical data format** `(HDF5) <http://extremecomputingtraining.anl.gov/files/2015/03/HDF5-Intro-aug7-130.pdf>`_ files (referred to as h5USID files) using python

* Jump to our `GitHub project <https://github.com/pycroscopy/pyNSID>`_

.. note::
   We are running weekly hackathons for pyNSID development every Friday from 3-5 PM - USA Eastern time.
   The requirements for participation are: knowledge of python, numpy, h5py, git, NSID rules.
   Please email vasudevanrk *at* ornl.gov to be added to the hackathons

Why?
-----
As we see it, there are a few opportunities in scientific research:

**1. Growing data sizes**
  * Need to be able to effortlessly accommodate datasets that are kB to TB and beyond
  * *Need: Scalable storage resources and compatible, scalable file structures*

**2. Increasing data and metadata complexity**
  * Sophisticated imaging and spectroscopy modes resulting in 5,6,7... dimensional data
  * *Need: Generalized data formatting and ability to store rich metadata accompanying central data*

**3. Multiple file formats**
  * Different formats from each instrument. Proprietary in most cases
  * Incompatible for and impeding correlation
  * *Need: Open, instrument-independent data format for storing and sharing data*

Who?
----
* We envision pyNSID to be a convenient package that facilitates all scientists to store and exchange data across scientific domains with ease.
* This project is being led by staff members at Oak Ridge National Laboratory (ORNL), and professors at University of Tennessee, Knoxville
* We invite anyone interested to join our team to build better and free software for the scientific community
* Please visit our `credits and acknowledgements <./credits.html>`_ page for more information.
* If you are interested in integrating our data model (NSID) with your existing package, please `get in touch <./contact.html>`_ with us.
