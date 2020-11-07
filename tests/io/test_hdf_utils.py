from __future__ import division, print_function, unicode_literals, absolute_import
import unittest
import sys
import h5py
import numpy as np
import dask.array as da
import tempfile
from sidpy.hdf.hdf_utils import write_simple_attrs

sys.path.append("../../")
import pyNSID


def make_simple_h5_dataset():
    """
    simple h5 dataset with dimesnsion arrays but not attached
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = tmp_dir + 'hdf5_simple.h5'
    h5_file = h5py.File(file_path, 'a')
    h5_group = h5_file.create_group('MyGroup')
    data = np.random.normal(size=(2, 3))
    h5_dataset = h5_group.create_dataset('data', data=data)

    dims = {0: h5_group.create_dataset('a', np.arange(data.shape[0])),
            1: h5_group.create_dataset('b', np.arange(data.shape[1]))}
    return h5_file


h5_simple_file = make_simple_h5_dataset()


def make_nsid_dataset_no_dim_attached():
    """
    except for the dimensions attached the h5 dataset which is fully pyNSID compatible
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = tmp_dir + 'nsid_simple.h5'
    h5_file = h5py.File(file_path, 'a')
    h5_group = h5_file.create_group('MyGroup')
    data = np.random.normal(size=(2, 3))
    h5_dataset = h5_group.create_dataset('data', data=data)
    attrs_to_write = {'quantity': 'quantity', 'units': 'units', 'nsid_version': 'version',
                      'main_data_name': 'title', 'data_type': 'data_type.name',
                      'modality': 'modality', 'source': 'test'}

    write_simple_attrs(h5_dataset, attrs_to_write)

    dims = {0: h5_group.create_dataset('a', data=np.arange(data.shape[0])),
            1: h5_group.create_dataset('b', data=np.arange(data.shape[1]))}
    for dim, this_dim_dset in dims.items():
        name = this_dim_dset.name.split('/')[-1]
        attrs_to_write = {'name': name, 'units': 'units', 'quantity': 'quantity',
                          'dimension_type': 'dimension_type.name', 'nsid_version': 'test'}

        write_simple_attrs(this_dim_dset, attrs_to_write)

        this_dim_dset.make_scale(name)
        h5_dataset.dims[dim].label = name
        # h5_dataset.dims[dim].attach_scale(this_dim_dset)
    return h5_file


def make_simple_nsid_dataset():
    """
    h5 dataset which is fully pyNSID compatible
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = tmp_dir + 'nsid_simple.h5'
    h5_file = h5py.File(file_path, 'a')
    h5_group = h5_file.create_group('MyGroup')
    data = np.random.normal(size=(2, 3))
    h5_dataset = h5_group.create_dataset('data', data=data)
    attrs_to_write = {'quantity': 'quantity', 'units': 'units', 'nsid_version': 'version',
                      'main_data_name': 'title', 'data_type': 'data_type.name',
                      'modality': 'modality', 'source': 'test'}

    write_simple_attrs(h5_dataset, attrs_to_write)

    dims = {0: h5_group.create_dataset('a', data=np.arange(data.shape[0])),
            1: h5_group.create_dataset('b', data=np.arange(data.shape[1]))}
    for dim, this_dim_dset in dims.items():
        name = this_dim_dset.name.split('/')[-1]
        attrs_to_write = {'name': name, 'units': 'units', 'quantity': 'quantity',
                          'dimension_type': 'dimension_type.name', 'nsid_version': 'test'}

        write_simple_attrs(this_dim_dset, attrs_to_write)

        this_dim_dset.make_scale(name)
        h5_dataset.dims[dim].label = name
        h5_dataset.dims[dim].attach_scale(this_dim_dset)
    return h5_file

def make_nsid_length_dim_wrong():
    """
    h5 dataset which is fully pyNSID compatible
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = tmp_dir + 'nsid_simple.h5'
    h5_file = h5py.File(file_path, 'a')
    h5_group = h5_file.create_group('MyGroup')
    data = np.random.normal(size=(2, 3))
    h5_dataset = h5_group.create_dataset('data', data=data)
    attrs_to_write = {'quantity': 'quantity', 'units': 'units', 'nsid_version': 'version',
                      'main_data_name': 'title', 'data_type': 'data_type.name',
                      'modality': 'modality', 'source': 'test'}

    write_simple_attrs(h5_dataset, attrs_to_write)

    dims = {0: h5_group.create_dataset('a', data=np.arange(data.shape[0]*3)),
            1: h5_group.create_dataset('b', data=np.arange(data.shape[1]))}
    for dim, this_dim_dset in dims.items():
        name = this_dim_dset.name.split('/')[-1]
        attrs_to_write = {'name': name, 'units': 'units', 'quantity': 'quantity',
                          'dimension_type': 'dimension_type.name', 'nsid_version': 'test'}

        write_simple_attrs(this_dim_dset, attrs_to_write)

        this_dim_dset.make_scale(name)
        h5_dataset.dims[dim].label = name
        h5_dataset.dims[dim].attach_scale(this_dim_dset)
    return h5_file


h5_nsid_no_dim_attached = make_nsid_dataset_no_dim_attached()
h5_nsid_simple = make_simple_nsid_dataset()
h5_nsid_wrong_dim_length = make_nsid_length_dim_wrong()

class TestGetAllMain(unittest.TestCase):

    def test_not_invalid_input(self):
        # Consider passing numpy arrays, HDF5 datasets
        # Expect TypeErrors
        pass

    def test_h5_file_instead_of_group(self):
        pass

    def test_one_main_dataset(self):
        pass

    def test_multiple_main_dsets_in_same_group(self):
        pass

    def test_multiple_main_dsets_in_diff_nested_groups(self):
        pass


class TestFindDataset(unittest.TestCase):
    # This function inherits a good portion of the code from sidpy.
    # We don't yet have the functionality to upconvert to sidpy.Dataset yet

    def test_one_not_main_dataset(self):
        # should return as h5py.Dataset
        pass

    def test_one_is_main_dataset(self):
        # should return the dataset as a sidpy.Dataset object
        pass

    def test_one_is_main_other_is_not(self):
        # Should return one as sidpy.Dataset and the other as h5py.Dataset
        pass


class TestCheckIfMain(unittest.TestCase):

    def test_not_h5_dataset(self):

        self.assertFalse(pyNSID.hdf_utils.check_if_main(np.arange(3)))

        self.assertFalse(pyNSID.hdf_utils.check_if_main(da.from_array(np.arange(3))))

        self.assertFalse(pyNSID.hdf_utils.check_if_main(h5_simple_file['MyGroup']))

        self.assertFalse(pyNSID.hdf_utils.check_if_main(h5_simple_file))

    def test_dims_missing(self):
        self.assertFalse(pyNSID.hdf_utils.check_if_main(h5_simple_file['MyGroup']['data']))

    def test_dim_exist_but_scales_not_attached_to_main(self):
        # test first same file without dimension attached
        self.assertFalse(pyNSID.hdf_utils.check_if_main(h5_nsid_no_dim_attached['MyGroup']['data']))

    def test_dim_sizes_not_matching_main(self):
        self.assertFalse(pyNSID.hdf_utils.check_if_main(h5_nsid_wrong_dim_length['MyGroup']['data']))


    def test_mandatory_attrs_not_present(self):
        for key in  ['quantity', 'units', 'main_data_name', 'data_type', 'modality', 'source']:
            attribute = h5_nsid_simple['MyGroup']['data'].attrs[key]

            del h5_nsid_simple['MyGroup']['data'].attrs[key]
            self.assertFalse(pyNSID.hdf_utils.check_if_main(h5_nsid_simple['MyGroup']['data']))
            h5_nsid_simple['MyGroup']['data'].attrs[key] = attribute

    def test_invalid_types_for_str_attrs(self):
        for key in ['quantity', 'units', 'main_data_name', 'data_type', 'modality', 'source']:
            attribute = h5_nsid_simple['MyGroup']['data'].attrs[key]
            h5_nsid_simple['MyGroup']['data'].attrs[key] = 1
            self.assertFalse(pyNSID.hdf_utils.check_if_main(h5_nsid_simple['MyGroup']['data']))
            h5_nsid_simple['MyGroup']['data'].attrs[key] = attribute

    def test_dset_is_main(self):
        self.assertTrue(pyNSID.hdf_utils.check_if_main(h5_nsid_simple['MyGroup']['data']))


class TestLinkAsMain(unittest.TestCase):
    # Perhaps this function could call the validate function
    # So some of these tests could actually be in the validate function

    def test_all_dims_in_mem(self):
        # All dimensions are sidpy.Dimension objects
        pass

    def test_all_dims_already_h5_datasets(self):
        pass

    def test_dims_and_h5_main_in_diff_files(self):
        pass

    def test_some_dims_in_mem_others_h5_dsets(self):
        pass

    def test_dim_in_mem_same_name_as_dim_in_h5(self):
        pass

    def test_dim_size_mismatch_main_shape(self):
        pass

    def test_too_few_dims(self):
        pass

    def test_too_many_dims(self):
        pass

    def test_h5_main_invalid_object_type(self):
        pass

    def test_dim_dict_invalid_obj_type(self):
        pass

    def test_items_in_dim_dict_invalid_obj_type(self):
        pass


class TestValidateMainDset(unittest.TestCase):

    def base_test(self):
        pass

    def test_imporper_h5_main_type(self):
        pass

    def test_must_be_h5_but_is_not(self):
        pass


if __name__ == '__main__':
    unittest.main()
