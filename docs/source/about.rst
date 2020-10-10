======
pyNSID
======

**Python framework for storing, visualizing, and processing spectroscopy, imaging or any observational / experimental data**

What?
------
* The `N-Dimensional Spectroscopic and Imaging Data (NSID) model <../nsid.html>`_:

  * facilitates the representation of most spectroscopic or imaging data regardless of their origin, modality, size, or dimensionality.
  * enables the development of instrument- and modality- agnostic data processing and analysis algorithms.
  * is just a definition or a blueprint rather than something tangible and readily usable.
  * cannot handle niche cases like spiral scans, compressed sensing, etc. given that these data do not have an N-dimensional form.
    However, our sister project - `pyUSID <../pyUSID/about.html>`_ was built to handle such complex scenarios
* pyNSID is a `python <http://www.python.org/>`_ package that currently provides three pieces of functionality:

  #. **io**: Primarily, it enables the storage and access of NSID in **hierarchical data format** `(HDF5) <http://extremecomputingtraining.anl.gov/files/2015/03/HDF5-Intro-aug7-130.pdf>`_ files (referred to as h5USID files) using python
  #. **viz**: It has handy tools for visualizing NSID and general scientific data
  #. **processing**: It provides a framework for formulating scientific problems into computational problems.
     See `pycroscopy <../pycroscopy/about.html>`_ - a sister project that uses pyNSID for analysis of microscopy data.

* Just as scipy uses numpy underneath, scientific packages like **pycroscopy** use **pyNSID** and **pyUSID** for all file-handling, data processing, and generating plots for journal publications
* pyNSID uses popular packages such as numpy, h5py, dask, matplotlib, etc. for most of the storage, computation, and visualization.
* **pyNSID is currently in the early stages of development**. The underlying code may change / be reorganized substantially.
* Jump to our `GitHub project <https://github.com/pycroscopy/pyNSID>`_

.. note::
   We are running weekly hackathons for pyNSID development every Friday from 3-5 PM - USA Eastern time.
   The requirements for participation are: knowledge of python, numpy, h5py, git, NSID rules.
   Please email vasudevanrk *at* ornl.gov to be added to the hackathons

Why?
-----
As we see it, there are a few opportunities in scientific imaging (that surely apply to several other scientific domains):

**1. Growing data sizes**
  * Cannot use desktop computers for analysis
  * *Need: High performance computing, storage resources and compatible, scalable file structures*

**2. Increasing data complexity**
  * Sophisticated imaging and spectroscopy modes resulting in 5,6,7... dimensional data
  * *Need: Robust software and generalized data formatting*

**3. Multiple file formats**
  * Different formats from each instrument. Proprietary in most cases
  * Incompatible for correlation
  * *Need: Open, instrument-independent data format*

**4. Expensive analysis software**
  * Software supplied with instruments often insufficient / incapable of custom analysis routines
  * Commercial software (Eg: Matlab, Origin..) are often prohibitively expensive.
  * *Need: Free, powerful, open source, user-friendly software*

**5. Closed science**
  * Analysis software and data not shared
  * No guarantees of reproducibility or traceability
  * *Need: open source data structures, file formats, centralized code and data repositories*

Who?
----
* We envision pyNSID to be a convenient package that facilitates all scientists to store and exchange data across scientific domains with ease.
* This project is being led by staff members at Oak Ridge National Laboratory (ORNL), and professors at University of Tennessee, Knoxville
* We invite anyone interested to join our team to build better, free software for the scientific community
* Please visit our `credits and acknowledgements <./credits.html>`_ page for more information.
* If you are interested in integrating our data model (NSID) with your existing package, please `get in touch <./contact.html>`_ with us.
