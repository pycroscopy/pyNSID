from __future__ import division, print_function, unicode_literals, absolute_import
import unittest
import sys
import h5py
import numpy as np
from os import remove

sys.path.insert(0, "../../../sidpy/")
import sidpy

sys.path.insert(0, "../../")
from pyNSID.io import hdf_io
import pyNSID

print(sidpy.__version__)


#Suhas
class TestCreateEmptyDataset(unittest.TestCase):

    def base_test(self, dims=3):
        h5_f = h5py.File('test_empty_dset.h5', 'w')
        h5_group = h5_f.create_group('MyGroup')
        shape = tuple([np.random.randint(low=2, high=6) for _ in range(dims)])
        dataset_name = 'test_dataset'
        empty_dset = hdf_io.create_empty_dataset(shape, h5_group, dataset_name)

        assert type(empty_dset) == h5py._hl.dataset.Dataset, "Output is not a h5py dataset"
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
            h5_group = list(tuple(2, 5), np.array([1, 30, 2]))
            _ = hdf_io.create_empty_dataset(shape, h5_group, dataset_name)
        h5_f.close()
        remove('test_empty.h5')

    def test_obj_with_same_name_already_exists_in_file(self):
        pass

    def test_name_kwarg_used_correctly(self):
        pass

#Gerd
class TestWriteNSIDataset(unittest.TestCase):
    def base_test(self, dims=3, dim_types=['spatial', 'spatial', 'spectral'],
                  data_type='complex', verbose=True):
        h5_f = h5py.File('test.h5', 'w')
        h5_group = h5_f.create_group('MyGroup')
        shape = tuple([np.random.randint(low=2, high=10) for _ in range(dims)])
        data = np.random.normal(size=shape)
        if data_type == 'complex':
            data = np.random.normal(size=tuple(shape)) + 1j * np.random.normal(size=tuple(shape))
        elif data_type == 'int':
            np.random.randint(low=0, high=1000, size=shape, dtype=np.int)
        elif data_type == 'float32':
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
                "Dimension name not correctly written, should be {} but is {} in file"\
                    .format(data_set._axes[ind].name, sidpy.hdf_utils.get_attr(h5_dset, 'DIMENSION_LABELS')[ind])

            assert sidpy.hdf_utils.get_attr(h5_dset, 'quantity') == data_set.quantity, \
                "Quantity attribute not correctly written, should be {} but is {} in file"\
                    .format(data_set.quantity, sidpy.hdf_utils.get_attr(h5_dset, 'quantity'))

            assert sidpy.hdf_utils.get_attr(h5_dset, 'source') == data_set.source, \
                "Source attribute not correctly written, should be {} but is {} in file"\
                    .format(data_set.source, sidpy.hdf_utils.get_attr(h5_dset, 'source'))

            assert sidpy.hdf_utils.get_attr(h5_dset, 'units') == data_set.units, \
                "Source attribute not correctly written, should be {} but is {} in file"\
                    .format(data_set.units, sidpy.hdf_utils.get_attr(h5_dset, 'units'))
        h5_f.close()
        remove('test.h5')

    def test_not_sidpy_dataset(self):
        h5_file = h5py.File('test.h5', 'w')
        h5_group = h5_file.create_group('MyGroup')

        numpy_array = np.arange(5)
        with self.assertRaises(TypeError):
            pyNSID.hdf_io.write_nsid_dataset(numpy_array, h5_group)

        string_input = 'nothing'
        with self.assertRaises(TypeError):
            pyNSID.hdf_io.write_nsid_dataset(string_input, h5_group)

        with self.assertRaises(TypeError):
            pyNSID.hdf_io.write_nsid_dataset(h5_group)

    def test_not_h5_group(self):
        data_set = sidpy.Dataset.from_array(np.zeros([5, 6]), name='Image')
        with self.assertRaises(TypeError):
            pyNSID.hdf_io.write_nsid_dataset(data_set)

        with self.assertRaises(TypeError):
            pyNSID.hdf_io.write_nsid_dataset(data_set, data_set)

        h5_file = h5py.File('test.h5', 'w')
        h5_group = h5_file.create_group('MyGroup')

        pyNSID.hdf_io.write_nsid_dataset(data_set, h5_group)

        with self.assertRaises(TypeError):
            pyNSID.hdf_io.write_nsid_dataset(data_set, np.zeros([5, 6]))

    def test_main_data_name_not_str(self):
        data_set = sidpy.Dataset.from_array(np.zeros([5, 6]), name='Image')
        h5_file = h5py.File('test.h5', 'w')
        h5_group = h5_file.create_group('MyGroup')

        with self.assertRaises(TypeError):
            pyNSID.hdf_io.write_nsid_dataset(data_set, h5_group, main_data_name=2)

    def test_main_data_name_given(self):
        data_set = sidpy.Dataset.from_array(np.zeros([5, 6]), name='Image')
        h5_file = h5py.File('test.h5', 'w')
        h5_group = h5_file.create_group('MyGroup')

        pyNSID.hdf_io.write_nsid_dataset(data_set, h5_group, main_data_name='good_name')
        self.assertTrue('good_name' in h5_group)

    def test_h5_file_in_read_only_mode(self):
        data_set = sidpy.Dataset.from_array(np.zeros([5, 6]), name='Image')
        h5_file = h5py.File('test.h5', 'r')

        with self.assertRaises(ValueError):
            pyNSID.hdf_io.write_nsid_dataset(data_set, h5_file)

    def test_h5_file_closed(self):
        data_set = sidpy.Dataset.from_array(np.zeros([5, 6]), name='Image')
        h5_file = h5py.File('test.h5', 'w')
        h5_group = h5_file.create_group('MyGroup')
        h5_file.close()

        with self.assertRaises(ValueError):
            pyNSID.hdf_io.write_nsid_dataset(data_set, h5_group)

    def test_group_already_has_obj_same_name_as_main_dset(self):
        data_set = sidpy.Dataset.from_array(np.zeros([5, 6]), name='Image')
        h5_file = h5py.File('test.h5', 'w')
        h5_group = h5_file.create_group('MyGroup')
        occupied = h5_group.create_group('occupied')
        with self.assertRaises(ValueError):
            pyNSID.hdf_io.write_nsid_dataset(data_set, h5_group,  main_data_name='occupied')

    def test_group_already_has_dim_h5_dset_diff_lengths(self):
        data_set = sidpy.Dataset.from_array(np.zeros([5, 6]), name='Image')
        h5_file = h5py.File('test.h5', 'w')
        h5_group = h5_file.create_group('MyGroup')

        pyNSID.hdf_io.write_nsid_dataset(data_set, h5_group)

        data_set = sidpy.Dataset.from_array(np.zeros([7, 8]), name='Image')
        with self.assertRaises(ValueError):
            pyNSID.hdf_io.write_nsid_dataset(data_set, h5_group)

    def test_group_already_has_dim_h5_dset_attrs_incorrect(self):
        pass

    def test_group_already_has_dim_h5_dset_correct(self):
        pass

    def test_complex_valued_main_dset(self):
        data_set = sidpy.Dataset.from_array(np.zeros([5, 6], dtype=complex), name='Image')
        h5_file = h5py.File('test.h5', 'w')
        h5_group = h5_file.create_group('MyGroup')

        pyNSID.hdf_io.write_nsid_dataset(data_set, h5_group)
        print(h5_group['Image']['Image'][()].dtype)
        self.assertTrue(h5_group['Image']['Image'][()].dtype == np.complex)

    def test_complex_valued_dimension(self):
        data_set = sidpy.Dataset.from_array(np.zeros([5, 6], dtype=complex), name='Image')
        h5_file = h5py.File('test.h5', 'w')
        h5_group = h5_file.create_group('MyGroup')
        data_set.set_dimension(0, sidpy.Dimension(np.arange(5, dtype=complex)))

        # with self.assertWarns('ComplexWarning'):
        pyNSID.hdf_io.write_nsid_dataset(data_set, h5_group)

    def test_book_keeping_attrs_written_to_group(self):
        shape = (10, 10, 15)
        data = np.random.normal(size=shape)
        h5_file = h5py.File('test2.h5', 'w')
        h5_group = h5_file.create_group('MyGroup')
        data_set = sidpy.Dataset.from_array(data[:], name='Image')
        sidpy.hdf_utils.write_book_keeping_attrs(h5_group)
        hdf_io.write_nsid_dataset(data_set, h5_group, main_data_name='data_1')
        for attr in ['machine_id', 'platform', 'sidpy_version', 'timestamp']:
            assert (attr in list(h5_group.attrs)) == True, \
                'book keeping attributes not correctly written, missing {}'.format(attr)
        h5_file.close()
        remove('test2.h5')

    def test_no_metadata(self):
        shape = (10, 10, 15)
        data = np.random.normal(size=shape)
        h5_file = h5py.File('test2.h5', 'w')
        h5_group = h5_file.create_group('MyGroup')
        data_set = sidpy.Dataset.from_array(data[:])
        sidpy.hdf_utils.write_book_keeping_attrs(h5_group)
        hdf_io.write_nsid_dataset(data_set, h5_group)
        h5_file.close()
        remove('test2.h5')

    def test_metadata_is_empty(self):
        data_set = sidpy.Dataset.from_array(np.zeros([5, 6]), name='Image')
        h5_file = h5py.File('test.h5', 'w')
        h5_group = h5_file.create_group('MyGroup')
        data_set.metadata = {}

        pyNSID.hdf_io.write_nsid_dataset(data_set, h5_group)

        self.assertFalse('metadata' in h5_group['Image'])

    def test_has_metadata_dict(self):
        data_set = sidpy.Dataset.from_array(np.zeros([5, 6]), name='Image')
        h5_file = h5py.File('test.h5', 'w')
        h5_group = h5_file.create_group('MyGroup')
        data_set.metadata = {'some': 'thing'}

        pyNSID.hdf_io.write_nsid_dataset(data_set, h5_group)

        self.assertTrue('metadata' in h5_group['Image'])

    # def test_metadata_not_dict(self):  # is actually tested in sidpy

    def test_metadata_is_nested(self):
        data_set = sidpy.Dataset.from_array(np.zeros([5, 6]), name='Image')
        h5_file = h5py.File('test.h5', 'w')
        h5_group = h5_file.create_group('MyGroup')
        data_set.metadata = {'some': {'some': 'thing'}}

        pyNSID.hdf_io.write_nsid_dataset(data_set, h5_group)

        self.assertTrue('metadata' in h5_group['Image'])
        print(h5_group['Image']['metadata'].attrs.keys())

        self.assertTrue(dict(h5_group['Image']['metadata'].attrs) == {'some-some': 'thing'})

    # def test_no_original_metadata(self): # cannot delete attribute delattr(data_set, 'original_metadata')

    def test_original_metadata_is_empty(self):
        data_set = sidpy.Dataset.from_array(np.zeros([5, 6]), name='Image')
        h5_file = h5py.File('test.h5', 'w')
        h5_group = h5_file.create_group('MyGroup')
        data_set.original_metadata = {}

        pyNSID.hdf_io.write_nsid_dataset(data_set, h5_group)

        self.assertFalse('original_metadata' in h5_group['Image'])

    def test_has_original_metadata_dict(self):
        data_set = sidpy.Dataset.from_array(np.zeros([5, 6]), name='Image')
        h5_file = h5py.File('test.h5', 'w')
        h5_group = h5_file.create_group('MyGroup')
        data_set.original_metadata = {'some': 'thing'}

        pyNSID.hdf_io.write_nsid_dataset(data_set, h5_group)

        self.assertTrue('original_metadata' in h5_group['Image'])

    # def test_original_metadata_not_dict(self): # tested in sidpy

    def test_original_metadata_is_nested(self):
        data_set = sidpy.Dataset.from_array(np.zeros([5, 6]), name='Image')
        h5_file = h5py.File('test.h5', 'w')
        h5_group = h5_file.create_group('MyGroup')
        data_set.original_metadata = {'some': {'some': 'thing'}}

        pyNSID.hdf_io.write_nsid_dataset(data_set, h5_group)

        self.assertTrue('original_metadata' in h5_group['Image'])
        self.assertTrue(dict(h5_group['Image']['original_metadata'].attrs) == {'some-some': 'thing'})

    # TODO check if datasets are indeed linked correctly to main

    def test_h5_dataset_property_of_sidpy_dataset_populated(self):
        data_set = sidpy.Dataset.from_array(np.zeros([5, 6]), name='Image')
        h5_file = h5py.File('test.h5', 'w')
        h5_group = h5_file.create_group('MyGroup')
        data_set.property = {'some': {'some': 'thing'}}

        pyNSID.hdf_io.write_nsid_dataset(data_set, h5_group)

        self.assertTrue('property' in h5_group['Image'])

    def test_dim_varied(self):
        for ind in range(1, 10):
            dim_types_base = ['spatial', 'spectral']
            data_types_base = ['float32', 'float64', 'int', 'complex']
            dim_types = [dim_types_base[np.random.randint(low=1, high=2)] for _ in range(ind)]
            for data_type in data_types_base:
                # TODO: Check what is wrong here
                # self.base_test(dims=ind, dim_types=dim_types, data_type=data_type)
                pass


class TestWriteResults(unittest.TestCase):

    def test_not_h5py_group_obj(self):
        #Set h5_group to be a list instead of an actual hdf5 group object
        h5_group = [10,115]
        shape = (5, 15, 16)
        data = np.random.randn(shape[0], shape[1], shape[2])
        data_set = sidpy.Dataset.from_array(data[:, :, :], name='Image')
        with self.assertRaises(TypeError):
            hdf_io.write_results(h5_group, dataset=data_set, attributes=None, process_name='TestProcess')

    def test_group_already_contains_objects_name_clashes(self):
        # Set h5_group to be a list instead of an actual hdf5 group object
        h5_file = h5py.File('test2.h5', 'w')
        h5_group = h5_file.create_group('MyGroup')
        shape = (5, 15, 16)
        data = np.random.randn(shape[0], shape[1], shape[2])
        data_set1 = sidpy.Dataset.from_array(data[:, :, :], name='Image')
        data_set2 = sidpy.Dataset.from_array(data[:, :, :], name='Image')
        hdf_io.write_results(h5_group, dataset=data_set1, attributes=None, process_name='TestProcess')

        #If we try to write data_set2 which has the same name, it probably should fail??
        #with self.assertRaises(ValueError):
        hdf_io.write_results(h5_group, dataset=data_set2, attributes=None, process_name='TestProcess')

    def test_no_sidpy_dataset_provided(self):
        h5_file = h5py.File('test2.h5', 'w')
        h5_group = h5_file.create_group('MyGroup')
        #going to pass a numpy array instead of a sidpy dataset
        with self.assertRaises(ValueError):
            hdf_io.write_results(h5_group, dataset=None, attributes=None, process_name='TestProcess')
        h5_file.close()
        remove('test2.h5')

    def test_not_a_sidpy_Dataset(self):
        shape = (10, 10, 15)
        data = np.random.normal(size=shape)
        h5_file = h5py.File('test2.h5', 'w')
        h5_group = h5_file.create_group('MyGroup')
        #should fail if standard numpy array is passed
        with self.assertRaises(ValueError):
            hdf_io.write_results(h5_group, dataset=data, attributes=None, process_name='TestProcess')
        h5_file.close()
        remove('test2.h5')

    def test_no_attributes_provided(self):
        shape = (10, 10, 15)
        data = np.random.normal(size=shape)
        h5_file = h5py.File('test2.h5', 'w')
        h5_group = h5_file.create_group('MyGroup')
        data_set = sidpy.Dataset.from_array(data[:, :, :], name='Image')

        # pass data without attributes
        hdf_io.write_results(h5_group, dataset=data_set, attributes=None, process_name='TestProcess')
        h5_file.close()
        remove('test2.h5')

    def test_attributes_not_dict(self):
        from collections import namedtuple
        shape = (10, 10, 15)
        data = np.random.normal(size=shape)
        h5_file = h5py.File('test2.h5', 'w')
        h5_group = h5_file.create_group('MyGroup')
        data_set = sidpy.Dataset.from_array(data[:, :, :], name='Image')
        attributes = namedtuple('Dimensions', ['x', 'y'])

        # pass data with attributes being something other than dictionary
        with self.assertRaises(TypeError):
            hdf_io.write_results(h5_group, dataset=data_set, attributes=attributes, process_name='TestProcess')
        h5_file.close()
        remove('test2.h5')

    def test_attributes_nested_dict(self):
        shape = (10, 10, 15)
        data = np.random.normal(size=shape)
        h5_file = h5py.File('test2.h5', 'w')
        h5_group = h5_file.create_group('MyGroup')
        data_set = sidpy.Dataset.from_array(data[:, :, :], name='Image')

        attributes = {'Key1':10, 'Key2': np.linspace(0,1,10), 'NestedDict': {'KeyLevel2_0':10,
                                                               'KeyLevel2_1': np.linspace(0,1,10)}}

        # pass data with nested dictionary, make sure it doesn't complain
        hdf_io.write_results(h5_group, dataset=data_set, attributes=attributes, process_name='TestProcess')

        h5_file.close()
        remove('test2.h5')

    def test_attributes_flat_dict(self):

        from collections.abc import MutableMapping
        # code to convert ini_dict to flattened dictionary
        # default seperater '_'

        def convert_flatten(d, parent_key='', sep='_'):
            items = []
            for k, v in d.items():
                new_key = parent_key + sep + k if parent_key else k

                if isinstance(v, MutableMapping):
                    items.extend(convert_flatten(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        shape = (10, 10, 15)
        data = np.random.normal(size=shape)
        h5_file = h5py.File('test2.h5', 'w')
        h5_group = h5_file.create_group('MyGroup')
        data_set = sidpy.Dataset.from_array(data[:, :, :], name='Image')

        attributes = {'Key1': 10, 'Key2': np.linspace(0, 1, 10), 'NestedDict': {'KeyLevel2_0': 10,
                                                                                'KeyLevel2_1': np.linspace(0, 1, 10)}}
        flattened_attributes = convert_flatten(attributes)
        # pass data with flattened dictionary, make sure it doesn't complain
        hdf_io.write_results(h5_group, dataset=data_set, attributes=flattened_attributes, process_name='TestProcess')

        h5_file.close()
        remove('test2.h5')

    def test_process_name_not_str(self):
        shape = (10, 10, 15)
        data = np.random.normal(size=shape)
        h5_file = h5py.File('test2.h5', 'w')
        h5_group = h5_file.create_group('MyGroup')
        data_set = sidpy.Dataset.from_array(data[:, :, :], name='Image')
        # pass data with process_name being something other than a string
        with self.assertRaises(TypeError):
            hdf_io.write_results(h5_group, dataset=data_set, attributes=None, process_name=['This shouldnt work'])
        h5_file.close()
        remove('test2.h5')

    def test_process_name_no_name_clashes(self):
        shape = (10, 10, 15)
        data = np.random.normal(size=shape)
        h5_file = h5py.File('test2.h5', 'w')
        h5_group = h5_file.create_group('MyGroup')
        data_set = sidpy.Dataset.from_array(data[:, :, :], name='Image')
        # pass data with process_name being something other than a string
        hdf_io.write_results(h5_group, dataset=data_set, attributes=None, process_name='This should work')
        h5_file.close()
        remove('test2.h5')

    def test_process_name_has_name_clashes(self):
        pass
        '''
        It is actually expected behavior to not fail when this happens
        so, skipping test.
        ----
        shape = (10, 10, 15)
        data = np.random.normal(size=shape)
        h5_file = h5py.File('test2.h5', 'w')
        h5_group = h5_file.create_group('MyGroup')
        data_set = sidpy.Dataset.from_array(data[:, :, :], name='Image')
        # pass data with process_name being something other than a string
        hdf_io.write_results(h5_group, dataset=data_set, attributes=None, process_name='This should work')
        #Trying to rewrite with same process name should give us an error, not sure which type though!
        #with self.assertRaises(TypeError):
        hdf_io.write_results(h5_group, dataset=data_set, attributes=None, process_name='This should work')

        sidpy.hdf_utils.print_tree(h5_file)
        assert False
        h5_file.close()

        remove('test2.h5')'''

    def test_multiple_sidpy_datasets_as_results(self):
        shape = (10, 10, 15)
        data = np.random.normal(size=shape)
        data2 = np.random.normal(size=shape)
        h5_file = h5py.File('test2.h5', 'w')
        h5_group = h5_file.create_group('MyGroup')

        data_set = sidpy.Dataset.from_array(data[:, :, :], name='Image')
        data_set2 = sidpy.Dataset.from_array(data2[:, :, :], name='Image2')

        results = [data_set, data_set2]

        hdf_io.write_results(h5_group, dataset=results, attributes=None, process_name='This should work')
        h5_file.close()
        remove('test2.h5')

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
