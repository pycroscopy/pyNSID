from __future__ import division, print_function, unicode_literals, absolute_import
import unittest
import sys
sys.path.append("../pyNSID/")


class TestImport(unittest.TestCase):

    def test_basic(self):
        import pyNSID as nsid
        print(nsid.__version__)
        self.assertTrue(True)

class TestWritingUtilities(unittest.TestCase):

    def test_create_empty_dataset(self):
        from pyNSID.io.hdf_io import create_empty_dataset
        import h5py
        from os import remove

        h5_f = h5py.File('test.h5', 'w')
        h5_group = h5_f.create_group('MyGroup')
        shape = (10,10,100)
        dataset_name = 'test_dataset'
        empty_dset = create_empty_dataset(shape, h5_group, dataset_name)

        assert type(empty_dset)==h5py._hl.dataset.Dataset, "Output is not a h5py dataset"
        assert empty_dset.shape == shape, "Output shape is {} but should be {}".format(empty_dset.shape, shape)

        #close file, delete
        h5_f.close()
        remove('test.h5')

    def test_create_nsid_dataset(self):
        from pyNSID.io.hdf_io import write_nsid_dataset
        import h5py
        import sidpy as sid
        import numpy as np

        h5_f = h5py.File('test.h5', 'w')
        h5_group = h5_f.create_group('MyGroup')
        shape = (10,10,32)
        data = np.random.randn(10,10,32)
        data_set = sid.Dataset.from_array(data[:,:,:], name='Image')
        data_set.set_dimension(0, sid.Dimension('x', np.linspace(0, 100E-6, shape[0]),
                                                units='a.u', quantity='x',
                                                dimension_type='spatial'))
        data_set.set_dimension(1, sid.Dimension('y', np.linspace(0, 200, shape[1]),
                                                units='a.u', quantity='y',
                                                dimension_type='spatial'))

        data_set.set_dimension(1, sid.Dimension('y', np.linspace(0, 200, shape[1]),
                                                units='a.u', quantity='y',
                                                dimension_type='spatial'))


        h5_dset = write_nsid_dataset(data_set, h5_group, main_data_name='test2', verbose=True)

        assert type(h5_dset) == h5py._hl.dataset.Dataset, "Output is not a h5py dataset"
        assert h5_dset.shape == shape, "Output shape is {} but should be {}".format(h5_dset.shape, shape)



