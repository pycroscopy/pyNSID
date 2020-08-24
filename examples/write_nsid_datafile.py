import numpy as np
import h5py
import sidpy as sid
import sys
sys.path.append('../pyNSID/')

import pyNSID as nsid

data_set = sid.Dataset.from_array(np.zeros([4, 5, 10]), name='zeros')
print(data_set)

###

h5_file = h5py.File("zeros.hf5")
if 'Measurement_000' in h5_file:
    del h5_file['Measurement_000/Channel_000']
h5_group = h5_file.create_group('Measurement_000/Channel_000')

h5_dataset = nsid. write_nsid_dataset(data_set, h5_group, main_data_name='')

sid.hdf.hdf_utils.print_tree(h5_file)