from __future__ import division, print_function, unicode_literals, absolute_import
import unittest
import sys
import h5py
import numpy as np
from os import remove
import sidpy

sys.path.append("../pyNSID/")
from pyNSID.io import hdf_io


class TestCreateEmptyDataset(unittest.TestCase):

    def base_test(self, dims = 3):
        h5_f = h5py.File('test_empty_dset.h5', 'w')
        h5_group = h5_f.create_group('MyGroup')
        shape = tuple([np.random.randint(low=2, high = 6) for _ in range(dims)])
        dataset_name = 'test_dataset'
        empty_dset = hdf_io.create_empty_dataset(shape, h5_group, dataset_name)

        assert type(empty_dset)==h5py._hl.dataset.Dataset, "Output is not a h5py dataset"
        assert empty_dset.shape == shape, "Output shape is {} but should be {}".format(empty_dset.shape, shape)

        # close file, delete
        h5_f.close()
        remove('test_empty_dset.h5')

    def test_just_main_data(self):
        for ind in range(1, 6):
            self.base_test(dims=ind)

    def test_shape_not_array_like(self):
        pass

    def test_non_int_shape(self):
        # Test that the correct errors are raised
        h5_f = h5py.File('test_empty.h5', 'w')
        h5_group = h5_f.create_group('MyGroup')
        shape = (1.0, 5.6, 3.5)
        dataset_name = 'test_dataset'

        with self.assertRaises(ValueError):
            _ = hdf_io.create_empty_dataset(shape, h5_group, dataset_name)

        h5_f.close()
        remove('test_empty.h5')

    def test_invaid_group_object(self):
        h5_f = h5py.File('test_empty.h5', 'w')
        dataset_name = 'test_dataset'
        shape = (10, 5, 1)
        with self.assertRaises(TypeError):
            h5_group = list(tuple(2,5), np.array([1,30,2]))
            _ = hdf_io.create_empty_dataset(shape, h5_group, dataset_name)
        h5_f.close()
        remove('test_empty.h5')

    def test_obj_with_same_name_already_exists_in_file(self):
        pass

    def test_name_kwarg_used_correctly(self):
        pass


class TestWriteNSIDataset(unittest.TestCase):
    def base_test(self, dims=3, dim_types = ['spatial', 'spatial', 'spectral'],
                  data_type = 'complex', verbose=True):
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

        data_set = sidpy.Dataset.from_array(data[:], name='Image')

        for ind in range(dims):
            data_set.set_dimension(ind, sidpy.Dimension(np.linspace(-2, 2, num=data_set.shape[ind], endpoint=True),
                                                    name='x'+str(ind), units='um', quantity='Length',
                                                    dimension_type=dim_types[ind]))
        data_set.units = 'nm'
        data_set.source = 'CypherEast2'
        data_set.quantity = 'Excaliburs'

        h5_dset = hdf_io.write_nsid_dataset(data_set, h5_group, main_data_name='test2', verbose=verbose)

        assert type(h5_dset) == h5py._hl.dataset.Dataset, "Output is not a h5py dataset"
        assert h5_dset.shape == shape, "Output shape is {} but should be {}".format(h5_dset.shape, shape)

        for ind in range(len(sidpy.hdf_utils.get_attr(h5_dset, 'DIMENSION_LABELS'))):

            assert sidpy.hdf_utils.get_attr(h5_dset, 'DIMENSION_LABELS')[ind] == data_set._axes[ind].name, \
                "Dimension name not correctly written, should be {} but is {} in file".format(data_set._axes[ind].name, sidpy.hdf_utils.get_attr(h5_dset, 'DIMENSION_LABELS')[ind])

            assert sidpy.hdf_utils.get_attr(h5_dset, 'quantity') == data_set.quantity, \
                "Quantity attribute not correctly written, should be {} but is {} in file".format(data_set.quantity, sidpy.hdf_utils.get_attr(h5_dset, 'quantity'))

            assert sidpy.hdf_utils.get_attr(h5_dset, 'source') == data_set.source, \
                "Source attribute not correctly written, should be {} but is {} in file".format(data_set.source,
                                                                                                  sidpy.hdf_utils.get_attr(
                                                                                                      h5_dset, 'source'))
            assert sidpy.hdf_utils.get_attr(h5_dset, 'units') == data_set.units, \
                "Source attribute not correctly written, should be {} but is {} in file".format(data_set.units,
                                                                                                sidpy.hdf_utils.get_attr(
                                                                                            h5_dset, 'units'))
        h5_f.close()
        remove('test.h5')


    def test_not_sidpy_dataset(self):
        shape = (10,10,15)
        data_set = np.random.normal(size=shape)
        h5_file = h5py.File('test2.h5', 'w')
        h5_group = h5_file.create_group('MyGroup')
        with self.assertRaises(TypeError):
            hdf_io.write_nsid_dataset(data_set, h5_group, main_data_name='test')
        h5_file.close()
        remove('test2.h5')

    def test_not_h5_group(self):
        shape = (10, 10, 15)
        data = np.random.normal(size=shape)
        h5_file = h5py.File('test2.h5', 'w')
        h5_group = {'My_dict':0}
        data_set = sidpy.Dataset.from_array(data[:], name='Image')
        with self.assertRaises(TypeError):
            hdf_io.write_nsid_dataset(data_set, h5_group, main_data_name='test')
        h5_file.close()
        remove('test2.h5')

    def test_main_data_name_not_str(self):
        shape = (10, 10, 15)
        data = np.random.normal(size=shape)
        h5_file = h5py.File('test2.h5', 'w')
        h5_group = h5_file.create_group('MyGroup')
        data_set = sidpy.Dataset.from_array(data[:], name='Image')
        with self.assertRaises(TypeError):
            hdf_io.write_nsid_dataset(data_set, h5_group, main_data_name=[12345])
        h5_file.close()
        remove('test2.h5')

    def test_main_data_name_given(self):
        shape = (10, 10, 15)
        data = np.random.normal(size=shape)
        h5_file = h5py.File('test2.h5', 'w')
        h5_group = h5_file.create_group('MyGroup')
        data_set = sidpy.Dataset.from_array(data[:], name='Image')
        maindataname = 'Main_Data'
        hdf_io.write_nsid_dataset(data_set, h5_group, main_data_name=maindataname)
        remove('test2.h5')

    def test_h5_file_in_read_only_mode(self):
        shape = (10, 10, 15)
        data = np.random.normal(size=shape)
        h5_file = h5py.File('test2.h5', 'w')
        h5_group = h5_file.create_group('MyGroup')
        data_set = sidpy.Dataset.from_array(data[:], name='Image')
        h5_file.close()
        h5_file = h5py.File('test2.h5', 'r')
        h5_group = h5_file['MyGroup']
        with self.assertRaises(ValueError):
            hdf_io.write_nsid_dataset(data_set, h5_group)
        remove('test2.h5')

    def test_h5_file_closed(self):
        shape = (10, 10, 15)
        data = np.random.normal(size=shape)
        h5_file = h5py.File('test2.h5', 'w')
        h5_group = h5_file.create_group('MyGroup')
        data_set = sidpy.Dataset.from_array(data[:], name='Image')
        h5_file.close()
        with self.assertRaises(ValueError):
            hdf_io.write_nsid_dataset(data_set, h5_group)
        remove('test2.h5')

    def test_group_already_has_obj_same_name_as_main_dset(self):
        shape = (10, 10, 15)
        data = np.random.normal(size=shape)
        h5_file = h5py.File('test2.h5', 'w')
        h5_group = h5_file.create_group('MyGroup')
        data_set = sidpy.Dataset.from_array(data[:], name='Image')
        hdf_io.write_nsid_dataset(data_set, h5_group, main_data_name  ='data_1')
        with self.assertRaises(ValueError):
            hdf_io.write_nsid_dataset(data_set, h5_group, main_data_name='data_1')

        remove('test2.h5')

    def test_group_already_has_dim_h5_dset_diff_lengths(self):
        pass

    def test_group_already_has_dim_h5_dset_attrs_incorrect(self):
        pass

    def test_group_already_has_dim_h5_dset_correct(self):
        pass

    def test_complex_valued_main_dset(self):
        pass

    def test_complex_valued_dimension(self):
        dims = 3
        dim_types = ['spatial', 'spatial', 'spectral']
        h5_f = h5py.File('test.h5', 'w')
        h5_group = h5_f.create_group('MyGroup')
        shape = tuple([np.random.randint(low=2, high=10) for _ in range(dims)])
        data = np.random.normal(size=tuple(shape)) + 1j * np.random.normal(size=tuple(shape))
        data_set = sidpy.Dataset.from_array(data[:], name='Image')

        for ind in range(dims):
            data_set.set_dimension(ind, sidpy.Dimension(np.linspace(-2, 2, num=data_set.shape[ind],
                                                                    endpoint=True) +
                                1j*np.linspace(-2, 2, num=data_set.shape[ind], endpoint=True) ,
                                name='x' + str(ind), units='um', quantity='Length',
                                                    dimension_type=dim_types[ind]))
        data_set.units = 'nm'
        data_set.source = 'CypherEast2'
        data_set.quantity = 'Excaliburs'

        h5_dset = hdf_io.write_nsid_dataset(data_set, h5_group, main_data_name='test2', verbose=False)
        h5_f.close()
        remove('test.h5')

    def test_book_keeping_attrs_written_to_group(self):
        pass

    def test_no_metadata(self):
        pass

    def test_metadata_is_empty(self):
        pass

    def test_has_metadata_dict(self):
        pass

    def test_metadata_not_dict(self):
        pass

    def test_no_original_metadata(self):
        pass

    def test_original_metadata_is_empty(self):
        pass

    def test_has_original_metadata_dict(self):
        pass

    def test_original_metadata_not_dict(self):
        pass

    def test_metadata_is_nested(self):
        pass

    def test_original_metadata_is_nested(self):
        pass

    # TODO check if datasets are indeed linked correctly to main

    def test_h5_dataset_property_of_sidpy_dataset_populated(self):
        pass

    def test_dim_varied(self):
        for ind in range(1, 10):
            dim_types_base = ['spatial', 'spectral']
            data_types_base = ['float32', 'float64', 'int', 'complex']
            dim_types = [dim_types_base[np.random.randint(low=1, high=2)] for _ in range(ind)]
            for data_type in data_types_base:
                self.base_test(dims=ind, dim_types=dim_types, data_type=data_type)

class TestWriteResults(unittest.TestCase):

    def test_not_h5py_group_obj(self):
        pass

    def test_group_already_contains_objects_name_clashes(self):
        pass

    def test_no_sidpy_dataset_provided(self):
        pass

    def test_not_a_sidpy_Dataset(self):
        pass

    def test_no_attributes_provided(self):
        pass

    def test_attributes_not_dict(self):
        pass

    def test_attributes_nested_dict(self):
        pass

    def test_attributes_flat_dict(self):
        pass

    def test_process_name_not_str(self):
        pass

    def test_process_name_no_name_clashes(self):
        pass

    def test_process_name_has_name_clashes(self):
        pass

    def test_multiple_sidpy_datasets_as_results(self):
        pass

    def test_simple(self):
        h5_f = h5py.File('test_write_results.h5', 'w')
        h5_group = h5_f.create_group('MyGroup')
        shape = (5, 15, 16)
        data = np.random.randn(shape[0], shape[1], shape[2])
        data_set = sidpy.Dataset.from_array(data[:, :, :], name='Image')
        hdf_io.write_results(h5_group, dataset=data_set, attributes=None, process_name='TestProcess')

        # TODO: Add some more assertions
        h5_f.close()
        remove('test_write_results.h5')
