"""
================================================================================
01. Writing a sid Dataset to file in NSID format
================================================================================

**Gerd Duscher**

08/24/2020

**this file shows how to store quickly store a sid Dataset to NSID format**
"""
########################################################################################################################
# Introduction
# -------------
# Saving a data and their metadata to file in a comprehensive way after acquisition, as intermediate or final results
# is at the core of any data analysis.
# The NSID data format is an attempt to meet those requirement as painless and universal as possible.
# In the following, we will create a sid.Dataset from a numpy array, which we will store as NSID format in its HDF5 file
########################################################################################################################

# Import numpy and h5py as the basis for the following operation
import numpy as np
import h5py

# All data analysis in pycroscopy is based on sid.Datasets
import sidpy as sid

# Utilize the NSID package for writing
import sys
sys.path.append('../pyNSID/')
import pyNSID as nsid

########################################################################################################################
# Making a sid Dataset (which is based on dask) is described in the sid Documentation
# Here, we just make a basic sid.Dataset from a numpy array

data_set = sid.Dataset.from_array(np.zeros([4, 5, 10]), name='zeros')
print(data_set)

########################################################################################################################
# Creating a HDF5 file and groups using h5py is described in the h5py_primer in this directory
# For testing reasons, we first delete the Channel_000 group

h5_file = h5py.File("zeros.hf5", mode='a')
if 'Measurement_000' in h5_file:
    del h5_file['Measurement_000/Channel_000']
h5_group = h5_file.create_group('Measurement_000/Channel_000')
########################################################################################################################

########################################################################################################################
# Write this sid.Dataset to file with one simple command
# We use the sid hdf_utilities to look at the created h5py file structure
#
# Please note that the NSID dataset has the dimensions (a,b,c) attached as attributes,
# which are accessible through "h5_dataset.dims". Look at hf5py for more information.
#
# The HDF55 group "original_metadata" contains contain all the information of the original file as a dictionary type
# in the attributes original_metadata.attrs (here empty)

h5_dataset = nsid.hdf_io.write_nsid_dataset(data_set, h5_group, main_data_name='zeros')

sid.hdf.hdf_utils.print_tree(h5_file)

print('dimension of hdf5 dataset: ', h5_dataset.dims)
print('name of hdf5 dataset: ', h5_dataset.name)

########################################################################################################################


########################################################################################################################
# Read NSID Dataset into sid.Dataset with two simple command
#
reader = nsid.NSIDReader(h5_group)
sid_datasets = reader.read()

# Let's see what we got
for i, dataset in enumerate(sid_datasets):
    print(dataset.title)
print('read sidpy dataset 1 - printing associated axis a: ', sid_datasets[0].a)
########################################################################################################################

########################################################################################################################
# we can also read any specific h5py dataset

dataset = reader.read_h5py_dataset(h5_group['zeros'])

print(dataset)

########################################################################################################################

########################################################################################################################
# A result can entail just some values or properties which are most effectivly stored in a dictionary.
# Alternatively, the results are another dataset, or both.
# Here we just add 1 to our dataset and write it to disc.

results = {'added': 1}
result_dataset = dataset.like_data(dataset+1)
result_dataset.title = 'ones'
result_dataset.source = dataset.title
print('source', result_dataset.source, dataset.title)

results_group = nsid.hdf_io.write_results(h5_group, dataset=result_dataset, attributes=results, process_name = 'add one')
print(results_group)

sid.hdf.hdf_utils.print_tree(h5_file)

########################################################################################################################

########################################################################################################################
# If we read the file again, we get an additional main dataset:
sid_datasets = reader.read()

# Let's see what we got
for i, dataset in enumerate(sid_datasets):
    print(dataset.title)

########################################################################################################################

########################################################################################################################
# At the end of our program, we need to close the h5py file.
# We cannot close it earlier in case the sidoy dataset is large and then will be only read on demand.

h5_group.file.close()
########################################################################################################################
