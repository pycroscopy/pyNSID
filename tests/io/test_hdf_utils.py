from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import sys
import unittest
from typing import Tuple, Type

import h5py
import numpy as np

sys.path.append("../pyNSID/")
from pyNSID.io.hdf_io import write_nsid_dataset
from pyNSID.io.hdf_utils import find_dataset, read_h5py_dataset
from sidpy import Dataset, Dimension


def create_h5group(h5f_name: str, h5g_name: str) -> Type[h5py.Group]:
    mode = 'r+' if os.path.exists(h5f_name) else 'w'
    h5_f = h5py.File(h5f_name, mode)
    h5_group = h5_f.create_group(h5g_name)
    return h5_group


def write_dummy_dset(hf_group: Type[h5py.Group], dims: Tuple[int],
                     main_name: str, **kwargs) -> None:
    dnames = kwargs.get("dnames", np.arange(len(dims)))
    dset = Dataset.from_array(
            np.random.random([*dims]), name="new")
    for i in range(len(dims)):
        dset.set_dimension(
            i, Dimension(np.arange(dset.shape[i]), "{}".format(str(dnames[i]))))
    write_nsid_dataset(
        dset, hf_group, main_data_name=main_name)


def get_dset(hf_name: str, h5g_name: str, dset_name: str) -> Type[h5py.Dataset]:
    h5group = create_h5group(hf_name, h5g_name)
    write_dummy_dset(h5group, (10, 10, 5), dset_name)
    hf = h5py.File(hf_name, 'r')
    dset = find_dataset(hf, dset_name)[0]
    return dset


class test_read_h5py_dataset(unittest.TestCase):

    def test_wrong_input_type(self) -> None:
        self.tearDown()
        dataset = create_h5group('test.hdf5', 'dataset')
        err_msg = 'can only read single Dataset'
        with self.assertRaises(TypeError) as context:
            _ = read_h5py_dataset(dataset)
        self.assertTrue(err_msg in str(context.exception))

    def test_hdf5_info(self) -> None:
        hf_name = "test.hdf5"
        dset = get_dset(hf_name, "g", "d")
        dset.attrs["title"] = "dset_name"
        dataset = read_h5py_dataset(dset)
        self.assertTrue(dataset.h5_filename == hf_name)

    def test_attrs_title(self) -> None:
        self.tearDown()
        dset = get_dset("test.hdf5", "g", "d")
        dset.attrs["title"] = "dset_name"
        dataset = read_h5py_dataset(dset)
        self.assertTrue(dataset.title == 'dset_name')

    def test_attrs_units(self) -> None:
        self.tearDown()
        dset1 = get_dset("test.hdf5", "g1", "d1")
        dset2 = get_dset("test.hdf5", "g2", "d2")
        dset2.attrs["units"] = 'nA'
        dataset1 = read_h5py_dataset(dset1)
        dataset2 = read_h5py_dataset(dset2)
        self.assertTrue(dataset1.units == 'generic')
        self.assertTrue(dataset2.units == 'nA')

    def test_attrs_quantity(self) -> None:
        self.tearDown()
        dset1 = get_dset("test.hdf5", "g1", "d1")
        dset2 = get_dset("test.hdf5", "g2", "d2")
        dset2.attrs["quantity"] = 'Current'
        dataset1 = read_h5py_dataset(dset1)
        dataset2 = read_h5py_dataset(dset2)
        self.assertTrue(dataset1.quantity == 'generic')
        self.assertTrue(dataset2.quantity == 'Current')

    def test_attrs_datatype(self) -> None:
        self.tearDown()
        dset1 = get_dset("test.hdf5", "g1", "d1")
        dset2 = get_dset("test.hdf5", "g2", "d2")
        dset2.attrs["data_type"] = 'SPECTRAL_IMAGE'
        dataset1 = read_h5py_dataset(dset1)
        dataset2 = read_h5py_dataset(dset2)
        self.assertTrue(dataset1.data_type.name == "UNKNOWN")
        self.assertTrue(dataset2.data_type.name == 'SPECTRAL_IMAGE')

    def test_attrs_modality(self) -> None:
        self.tearDown()
        dset1 = get_dset("test.hdf5", "g1", "d1")
        dset2 = get_dset("test.hdf5", "g2", "d2")
        dset2.attrs["modality"] = 'modality'
        dataset1 = read_h5py_dataset(dset1)
        dataset2 = read_h5py_dataset(dset2)
        self.assertTrue(dataset1.modality == 'generic')
        self.assertTrue(dataset2.modality == 'modality')

    def test_attrs_source(self) -> None:
        self.tearDown()
        dset1 = get_dset("test.hdf5", "g1", "d1")
        dset2 = get_dset("test.hdf5", "g2", "d2")
        dset2.attrs["source"] = 'source'
        dataset1 = read_h5py_dataset(dset1)
        dataset2 = read_h5py_dataset(dset2)
        self.assertTrue(dataset1.source == 'generic')
        self.assertTrue(dataset2.source == 'source')

    def test_dims(self) -> None:
        self.tearDown()
        dset = get_dset("test.hdf5", "g", "d")
        dataset = read_h5py_dataset(dset)
        self.assertTrue(dataset._axes[0].name == '0')
        self.assertTrue(dataset._axes[1].name == '1')
        self.assertTrue(dataset._axes[2].name == '2')

    def test_all_attrs_inheritance(self) -> None:
        self.tearDown()
        dset = get_dset("test.hdf5", "g", "d")
        dataset = read_h5py_dataset(dset)
        self.assertTrue(all([v1 == v2 for (v1, v2) in
                            zip(dset.attrs.values(), dataset.attrs.values())][2:]))
        self.assertTrue(all([k1 == k2 for (k1, k2) in
                            zip(dset.attrs.keys(), dataset.attrs.keys())]))

    def tearDown(self, fname: str = 'test.hdf5') -> None:
        if os.path.exists(fname):
            os.remove(fname)


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
        pass

    def test_dims_missing(self):
        pass

    def test_dim_exist_but_scales_not_attached_to_main(self):
        pass

    def test_dim_sizes_not_matching_main(self):
        pass

    def test_mandatory_attrs_not_present(self):
        pass

    def test_invalid_types_for_str_attrs(self):
        pass

    def test_dset_is_main(self):
        pass


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
