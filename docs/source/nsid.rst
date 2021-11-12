NSID Specifications
===================
**Suhas Somnath**

A data collection
-----------------
Though it is best to refer to this concept as a ``dataset``, this concept is being referred to as a collection to avoid confusion with HDF5's ``dataset``
Fundamentally, any N-dimensional data collection (should) have the following key components:

* Central ``N``-dimensional array or dataset of interest.
  A `dimension` here refers to an independent variable against which data is collected.
  For example, in the case of a grayscale photograph, the `X` and `Y` axes are the independent variables or `dimensions` while the brightness values at each pixel constitute the central data array.
* Axes for each of the ``N`` dimensions. These would be individual values for each of the ``N`` variables against which data were collected in the central dataset.
  **Note**: These axes need to be linear. In other words, it is not necessary that the independent variable be varied linearly.
  For example, an independent variable like voltage or bias could be varied sinusoidally or as a bi-polar triangular or sawtooth waveform.
* Metadata that provides contextual information about this data collection.
  These could be measurement, observation, simulation, or analysis parameters necessary to provide a wholesome picture of the data collection.

N-Dimensional Spectroscopy and Imaging Data (NSID)
--------------------------------------------------
The core philosophy of NSID is to enable the storage of one or more data collections in a single file.
NSID was designed with Hierarchical Data Format (HDF5) files in mind though it may indeed be possible to implement the basic philosophy in other file formats as well.

In NSID, the central data is referred to as the ``Main Dataset`` and the axes are referred to as ``Ancillary Datasets``. Attributes attached to both ``Main`` and ``Ancillary``
datasets will be used to identify them as ``Main`` and ``Ancillary`` datasets among other ordinary HDF5 datasets in a file.

Group
~~~~~
It is strongly encouraged that all components (``Main dataset``, ``Ancillary dataset``, ``metadata``) associated with a single data collection be stored within a single HDF5 group
for the sake of cleanliness and avoiding confusion with another data collection.

Main Dataset
~~~~~~~~~~~~
The central data will be stored in a HDF5 dataset with the following properties:

* **name**: Arbitrary - No restrictions applied here so long as the name or title is unique within a given HDF5 group.
* **shape**: Arbitrary - matching the dimensionality of the data
* **dtype**: basic types like ``int``, ``float``, and ``complex`` only.
  Though HDF5 and h5py can indeed store compound valued datasets,
  it is not recommended that such complex dtypes be used since other languages like Fortran or Java may be unable to read such dtypes.
  Therefore, such data should be broken up into independent datasets with simple dtypes.
* **chunks**: Leave as default / do not specify anything.
* **compression**: Preferably do not use anything. If compression is indeed necessary, consider using `gzip`.
* **Dimension scales**: Every single ``dimension`` needs to have at least one ``Dimension Scale`` attached to it with the name(s) of the dimension(s) as the `label` (s) for the scale.
  Normally, we would only have one dataset attached to each dimension.
* **Recommended Attributes**: Aligned with the specifications for `sidpy.Dataset <https://pycroscopy.github.io/sidpy/_autosummary/sidpy.sid.dataset.Dataset.html#sidpy.sid.dataset.Dataset>`_:

  * ``title``: `string`: Title for this dataset. Perhaps important if there are multiple datasets within the same HDF5 file.
  * ``quantity``: `string`: Physical quantity that is contained in this dataset
  * ``units``: `string`: Units for this physical quantity
  * ``data_type``: `string` : What kind of data this is.
    Example - image, image stack, video, hyperspectral image, etc.
  * ``modality``: `string` : Experimental / simulation modality - scientific meaning of data.
    Example - photograph, TEM micrograph, SPM Force-Distance spectroscopy.
  * ``source``: `string` : Source for dataset like the kind of instrument.
    One could go very deep here into either the algorithmic details (if this is a result from analysis)
    or the exact configurations for the instrument that generated this data collection.
    We are inclined on removing this attribute and having this concept expressed in the metadata alone.
  * ``nsid_version``: `string`: Version of the abstract NSID model.
    Currently, the pyNSID version is being used in place of the revision number for the NSID specification.

Ancillary Datasets
~~~~~~~~~~~~~~~~~~
Each of the `N` dimensions corresponding to the `N`-dimensional `Main Dataset` would be an independent HDF5 dataset with the following properties:

* **shape** - 1D only
* **dtype** - Simple data types like ``int``, ``float``, ``complex``
* **Required attributes** -

  * ``quantity``: `string`: Physical quantity that is contained in this dataset
  * ``units``: `string`: units for the physical quantity
  * ``dimension_type``: `string`: Kind of dimension - 'position', 'spectral', 'reciprocal'

The ancillary dataset for each dimension would be `attached <http://docs.h5py.org/en/stable/high/dims.html>`_ to the ``Main Dataset`` using
HDF5 `Dimension Scales <https://support.hdfgroup.org/HDF5/Tutor/h5dimscale.html>`_.


Metadata
~~~~~~~~
Contextual information will be mostly be broken down into two categories or variables just as in the ``sidpy.Dataset``
and contents each of these variables will be written in to HDF5 Groups of the same names:

* ``metadata`` - which will contain the most frequently used / important metadata
* ``original_metadata`` - raw metadata obtained from vendor's proprietary data files.

Mirror the (potentially) hierarchical or nested metadata into identical hierarchical HDF5 groups within the same parent HDF5 Group as the ``Main Dataset``.
This requires feedback from experts in schemas and ontologies.

Multiple data collections in same file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A single HDF5 file can contain multiple HDF5 datasets.
It is not necessary that all datasets be NSID-specific.
Similarly, the hierarchical nature of HDF5 will allow the storage of multiple NSID data collections within the same HDF5 file.
Strict restrictions will not be placed on how the datasets should be arranged.
Users are free to use and are encouraged to use the similar guidelines of `Measurement Groups <https://pycroscopy.github.io/USID/h5_usid.html#measurement-data>`_ and `Channels <https://pycroscopy.github.io/USID/usid_model.html#channels>`_ as defined in USID.

Data processing results in same file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We defined a `possible solution <https://pycroscopy.github.io/USID/h5_usid.html#tool-analysis-processing>`_ for capturing provenance between the source dataset and the results datasets.
Briefly, results would be stored in a group whose name would be formatted as ``SourceDataset-ProcessingAlgorithmName_NumericIndex``.
However, this solution does not work *elegantly* for certain situations:

* if multiple source datasets were used to produce a set of results datasets.
* if results are written into a different file.
* In general, the algorithm name was loosely defined.

Do get in touch if you have a better solution

Existing solutions
~~~~~~~~~~~~~~~~~~

Why not just use h5py?
----------------------
``h5py``, on top of which ``pyNSID`` is built, does indeed provide all the functionality necessary to support ``NSID``.
However, a layer of *convenience* and *standardization* is still useful / necessary for few reasons:

1. To ensure that data (in memory) are always stored in the `same standardized manner <https://pycroscopy.github.io/sidpy/_autosummary/sidpy.sid.dataset.Dataset.html#sidpy.sid.dataset.Dataset>`_.
   This is handled by ``sidpy.Dataset`` via ``SciFiReaders.NSIDReader``.
2. To make it ``easier to access relevant information <https://pycroscopy.github.io/sidpy/_autosummary/sidpy.sid.dataset.Dataset.html#sidpy.sid.dataset.Dataset>`_ from HDF5 datasets such as the dimensions, units, scales, etc. without needing to write a lot of h5py code.
3. To simplify certain ancillary tasks like identify all NSID datasets in a given file, seamlessly reusing datasets representing dimensions / copying datasets, verifying whether a dataset is indeed NSID or not.
4. To facilitate embarrassingly parallel computations on datasets along the lines of `pyUSID.Process <https://pycroscopy.github.io/pyUSID/auto_examples/intermediate/plot_process.html#sphx-glr-auto-examples-intermediate-plot-process-py>`_ without having to customize algorithms to a specific instrument format, etc.

From scientific literature
--------------------------
As of this writing we are aware of similar efforts:

* Multi-vendor consortium headed by UIUC - `electron microscopy data <https://emdatasets.com/format/>`_
* DREAM.3D - "`MXA: a customizable HDF5-based data format for multi-dimensional data sets <https://iopscience.iop.org/article/10.1088/0965-0393/18/6/065008>`_" by Michael Jackson
* APS at Argonne - "`Scientific data exchange: a schema for HDF5-based storage of raw and analyzed data <https://onlinelibrary.wiley.com/doi/full/10.1107/S160057751401604X?sentby=iucr>`_" by Francesco de Carlo.

However, all of these were targeting a specific scientific sub-domain / modality. They were not as simple / general as pyNSID.
