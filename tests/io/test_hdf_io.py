from __future__ import division, print_function, unicode_literals, absolute_import
import unittest
import sys
sys.path.append("../pyNSID/")


class TestCreateEmptyDataset(unittest.TestCase):

    def base_test(self, dims = 3):
        from pyNSID.io.hdf_io import create_empty_dataset
        import h5py
        import numpy as np
        from os import remove

        h5_f = h5py.File('test_empty_dset.h5', 'w')
        h5_group = h5_f.create_group('MyGroup')
        shape = tuple([np.random.randint(low=2, high = 6) for _ in range(dims)])
        dataset_name = 'test_dataset'
        empty_dset = create_empty_dataset(shape, h5_group, dataset_name)

        assert type(empty_dset)==h5py._hl.dataset.Dataset, "Output is not a h5py dataset"
        assert empty_dset.shape == shape, "Output shape is {} but should be {}".format(empty_dset.shape, shape)

        # close file, delete
        h5_f.close()
        remove('test_empty_dset.h5')

    def test_just_main_data(self):
        for ind in range(1,6):
            self.base_test(dims=ind)

    def test_invalid_shape(self):
        # Test that the correct errors are raised
        from pyNSID.io.hdf_io import create_empty_dataset
        import h5py
        import numpy as np
        from os import remove

        h5_f = h5py.File('test_empty.h5', 'w')
        h5_group = h5_f.create_group('MyGroup')
        shape = (1.0, 5.6, 3.5)
        dataset_name = 'test_dataset'

        with self.assertRaises(ValueError):
            _ = create_empty_dataset(shape, h5_group, dataset_name)

        h5_f.close()
        remove('test_empty.h5')

    def test_invaid_group_object(self):
        from pyNSID.io.hdf_io import create_empty_dataset
        import h5py
        from os import remove
        import numpy as np

        h5_f = h5py.File('test_empty.h5', 'w')
        dataset_name = 'test_dataset'
        shape = (10, 5, 1)
        with self.assertRaises(TypeError):
            h5_group = list(tuple(2,5), np.array([1,30,2]))
            _ = create_empty_dataset(shape, h5_group, dataset_name)
        h5_f.close()
        remove('test_empty.h5')


class TestWriteNSIDataset(unittest.TestCase):

    def test_simple(self):
        import numpy as np
        for ind in range(1,10):
            dim_types_base = ['spatial', 'spectral']
            data_types_base = ['float32', 'float64', 'int', 'complex']
            dim_types = [dim_types_base[np.random.randint(low=1, high=2)] for _ in range(ind)]
            for data_type in data_types_base:
                self.base_test_write_nsid_dataset(dims=ind, dim_types=dim_types, data_type=data_type)

    def base_test(self, dims=3, dim_types = ['spatial', 'spatial', 'spectral'],
                  data_type = 'complex', verbose=True):
        from pyNSID.io.hdf_io import write_nsid_dataset
        import h5py
        import sidpy as sid
        import numpy as np
        from os import remove, path

        h5_f = h5py.File('test.h5', 'w')
        h5_group = h5_f.create_group('MyGroup')
        shape = tuple([np.random.randint(low=2, high = 10) for _ in range(dims)])
        data = np.random.normal(size=shape)
        if data_type=='complex':
            data = np.random.normal(size=tuple(shape)) + 1j* np.random.normal(size=tuple(shape))
        elif data_type =='int':
            np.random.randint(low=0, high = 1000, size=shape, dtype = np.int)
        elif data_type =='float32':
            data = np.random.normal(size=shape)
            data = np.squeeze(np.array(data, dtype=np.float32))
        else:
            data = np.random.normal(size=shape)
            data = np.squeeze(np.array(data, dtype=np.float64))

        data_set = sid.Dataset.from_array(data[:], name='Image')

        for ind in range(dims):
            data_set.set_dimension(ind, sid.Dimension(np.linspace(-2, 2, num=data_set.shape[ind], endpoint=True),
                                                    name='x'+str(ind), units='um', quantity='Length',
                                                    dimension_type=dim_types[ind]))
        data_set.units = 'nm'
        data_set.source = 'CypherEast2'
        data_set.quantity = 'Excaliburs'

        h5_dset = write_nsid_dataset(data_set, h5_group, main_data_name='test2', verbose=verbose)

        assert type(h5_dset) == h5py._hl.dataset.Dataset, "Output is not a h5py dataset"
        assert h5_dset.shape == shape, "Output shape is {} but should be {}".format(h5_dset.shape, shape)

        for ind in range(len(sid.hdf_utils.get_attr(h5_dset, 'DIMENSION_LABELS'))):

            assert sid.hdf_utils.get_attr(h5_dset, 'DIMENSION_LABELS')[ind] == data_set._axes[ind].name, \
                "Dimension name not correctly written, should be {} but is {} in file".format(data_set._axes[ind].name, sid.hdf_utils.get_attr(h5_dset, 'DIMENSION_LABELS')[ind])

            assert sid.hdf_utils.get_attr(h5_dset, 'quantity') == data_set.quantity, \
                "Quantity attribute not correctly written, should be {} but is {} in file".format(data_set.quantity, sid.hdf_utils.get_attr(h5_dset, 'quantity'))

            assert sid.hdf_utils.get_attr(h5_dset, 'source') == data_set.source, \
                "Source attribute not correctly written, should be {} but is {} in file".format(data_set.source,
                                                                                                  sid.hdf_utils.get_attr(
                                                                                                      h5_dset, 'source'))
            assert sid.hdf_utils.get_attr(h5_dset, 'units') == data_set.units, \
                "Source attribute not correctly written, should be {} but is {} in file".format(data_set.units,
                                                                                                sid.hdf_utils.get_attr(
                                                                                            h5_dset, 'units'))
        h5_f.close()
        remove('test.h5')


class TestWriteResults(unittest.TestCase):

    def test_simple(self):
        from pyNSID.io.hdf_io import write_results
        import h5py
        import sidpy as sid
        import numpy as np
        from os import remove

        h5_f = h5py.File('test_write_results.h5', 'w')
        h5_group = h5_f.create_group('MyGroup')
        shape = (5, 15, 16)
        data = np.random.randn(shape[0], shape[1], shape[2])
        data_set = sid.Dataset.from_array(data[:, :, :], name='Image')
        write_results(h5_group, dataset=data_set, attributes=None, process_name='TestProcess')

        #TODO: Add some more assertions
        h5_f.close()
        remove('test_write_results.h5')
